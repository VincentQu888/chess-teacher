import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import wP from "./assets/pieces/wP.svg";
import wN from "./assets/pieces/wN.svg";
import wB from "./assets/pieces/wB.svg";
import wR from "./assets/pieces/wR.svg";
import wQ from "./assets/pieces/wQ.svg";
import wK from "./assets/pieces/wK.svg";
import bP from "./assets/pieces/bP.svg";
import bN from "./assets/pieces/bN.svg";
import bB from "./assets/pieces/bB.svg";
import bR from "./assets/pieces/bR.svg";
import bQ from "./assets/pieces/bQ.svg";
import bK from "./assets/pieces/bK.svg";

const DEFAULT_FEN =
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const DEFAULT_PROMPT = "";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["8", "7", "6", "5", "4", "3", "2", "1"];

const PIECE_IMAGES = {
  P: wP,
  N: wN,
  B: wB,
  R: wR,
  Q: wQ,
  K: wK,
  p: bP,
  n: bN,
  b: bB,
  r: bR,
  q: bQ,
  k: bK,
};

function parseFen(fen) {
  if (!fen) return Array.from({ length: 8 }, () => Array(8).fill(""));
  const boardPart = fen.split(" ")[0];
  const rows = boardPart.split("/");
  if (rows.length !== 8) {
    return Array.from({ length: 8 }, () => Array(8).fill(""));
  }

  return rows.map((row) => {
    const cells = [];
    for (const char of row) {
      if (/[1-8]/.test(char)) {
        const count = Number(char);
        for (let i = 0; i < count; i += 1) {
          cells.push("");
        }
      } else {
        cells.push(char);
      }
    }
    while (cells.length < 8) {
      cells.push("");
    }
    return cells.slice(0, 8);
  });
}

function pawnCountError(fen) {
  if (!fen) return "";
  const boardPart = fen.split(" ")[0] || "";
  let whitePawns = 0;
  let blackPawns = 0;
  for (const char of boardPart) {
    if (char === "P") whitePawns += 1;
    if (char === "p") blackPawns += 1;
  }
  if (whitePawns > 8 || blackPawns > 8) {
    return `Too many pawns (${whitePawns} white, ${blackPawns} black). Max 8 per side.`;
  }
  return "";
}

function formatMoveLine(line) {
  const score = line.formatted_score || "?";
  const moves = line.moves?.join(" ") || "";
  return `${score}  ${moves}`.trim();
}

// Eval-bar helpers. Both evals are stored from White's perspective so a single
// bottom-up fill represents White's share of the bar.
function stockfishWhitePct(score) {
  if (!score) return 50;
  if (score.mate != null) return score.mate > 0 ? 100 : 0;
  // Logistic squash of centipawns -> a smooth, lichess-like bar.
  return 100 / (1 + Math.exp(-(score.cp || 0) / 350));
}

function alphazeroWhitePct(valueWhite) {
  if (valueWhite == null) return 50;
  return ((valueWhite + 1) / 2) * 100;
}

function formatSf(score) {
  if (!score) return "\u2013";
  if (score.mate != null) return (score.mate > 0 ? "M" : "-M") + Math.abs(score.mate);
  const p = (score.cp || 0) / 100;
  return (p >= 0 ? "+" : "") + p.toFixed(1);
}

function formatAz(valueWhite) {
  if (valueWhite == null) return "\u2013";
  return (valueWhite >= 0 ? "+" : "") + valueWhite.toFixed(2);
}

function squareCenter(square) {
  const fileIndex = FILES.indexOf(square[0]);
  const rankIndex = 8 - Number(square[1]);
  return {
    x: (fileIndex + 0.5) * 12.5,
    y: (rankIndex + 0.5) * 12.5,
  };
}

export default function App() {
  const gameRef = useRef(new Chess(DEFAULT_FEN));
  const [gameFen, setGameFen] = useState(gameRef.current.fen());
  const [fenInput, setFenInput] = useState(DEFAULT_FEN);
  const [fenError, setFenError] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [result, setResult] = useState(null);
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [followUp, setFollowUp] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [activeSquare, setActiveSquare] = useState(null);
  const [legalTargets, setLegalTargets] = useState([]);
  const [dragFrom, setDragFrom] = useState(null);
  const [arrows, setArrows] = useState([]);
  const [arrowStart, setArrowStart] = useState(null);
  const [arrowPreview, setArrowPreview] = useState(null);
  const [highlights, setHighlights] = useState([]);
  const [showSaliency, setShowSaliency] = useState(false);
  const [saliency, setSaliency] = useState(null);
  const [saliencyMode, setSaliencyMode] = useState("value");
  const [saliencyLoading, setSaliencyLoading] = useState(false);

  const board = useMemo(() => parseFen(gameFen), [gameFen]);

  // Fetch neural-net attention saliency for the current position when the heatmap
  // is toggled on (and refresh it whenever the position changes).
  useEffect(() => {
    if (!showSaliency) {
      setSaliency(null);
      return;
    }
    let cancelled = false;
    setSaliencyLoading(true);
    fetch("http://localhost:8001/saliency", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fen: gameRef.current.fen() }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!cancelled) setSaliency(data);
      })
      .catch(() => {
        if (!cancelled) setSaliency(null);
      })
      .finally(() => {
        if (!cancelled) setSaliencyLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [showSaliency, gameFen, saliencyMode]);

  // Live dual eval bars (Stockfish + AlphaZero). Debounced so rapid moves don't
  // spawn a request per intermediate position. Fails silently if the API is down.
  useEffect(() => {
    let cancelled = false;
    const id = setTimeout(() => {
      fetch("http://localhost:8001/eval", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: gameRef.current.fen() }),
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => {
          if (!cancelled && data) setEvalData(data);
        })
        .catch(() => {});
    }, 200);
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [gameFen]);

  const syncFen = (nextFen) => {
    const trimmedFen = nextFen.trim();
    try {
      const nextGame = new Chess();
      nextGame.load(trimmedFen);

      const pawnError = pawnCountError(trimmedFen);
      if (pawnError) {
        setFenError(pawnError);
        return false;
      }

      gameRef.current = nextGame;
      const updated = nextGame.fen();
      setGameFen(updated);
      setFenInput(updated);
      setFenError("");
      setActiveSquare(null);
      setLegalTargets([]);
      return true;
    } catch (err) {
      setFenError("Invalid FEN");
      return false;
    }
  };

  const setTargetsForSquare = (square) => {
    const moves = gameRef.current.moves({ square, verbose: true });
    setLegalTargets(moves.map((move) => move.to));
  };

  const attemptMove = (from, to) => {
    const piece = gameRef.current.get(from);
    if (!piece) {
      return false;
    }
    const isPromotion =
      piece.type === "p" && (to[1] === "8" || to[1] === "1");
    const move = gameRef.current.move({
      from,
      to,
      promotion: isPromotion ? "q" : undefined,
    });
    if (!move) {
      return false;
    }
    const nextFen = gameRef.current.fen();
    setGameFen(nextFen);
    setFenInput(nextFen);
    setActiveSquare(null);
    setLegalTargets([]);
    return true;
  };

  const handleSquareClick = (square) => {
    const piece = gameRef.current.get(square);
    if (activeSquare) {
      if (square === activeSquare) {
        setActiveSquare(null);
        setLegalTargets([]);
        return;
      }
      if (attemptMove(activeSquare, square)) {
        return;
      }
    }
    if (piece && piece.color === gameRef.current.turn()) {
      setActiveSquare(square);
      setTargetsForSquare(square);
    } else {
      setActiveSquare(null);
      setLegalTargets([]);
    }
  };

  const handleDragStart = (event, square) => {
    const piece = gameRef.current.get(square);
    if (!piece || piece.color !== gameRef.current.turn()) {
      event.preventDefault();
      return;
    }
    setDragFrom(square);
  };

  const handleDrop = (event, square) => {
    event.preventDefault();
    if (!dragFrom) {
      return;
    }
    attemptMove(dragFrom, square);
    setDragFrom(null);
  };

  const clearAnnotations = () => {
    setArrows([]);
    setHighlights([]);
    setArrowStart(null);
    setArrowPreview(null);
  };

  const handleSquareMouseDown = (event, square) => {
    if (event.button === 0) {
      // Left mousedown: chess.com-style clear of arrows + red highlights.
      // Runs before onClick / dragstart so move logic and selection still work.
      clearAnnotations();
      return;
    }
    if (event.button === 2) {
      event.preventDefault();
      setArrowStart(square);
      setArrowPreview(null);
    }
  };

  const handleArrowEnd = (event, square) => {
    if (event.button !== 2) {
      return;
    }
    event.preventDefault();
    if (!arrowStart) {
      setArrowPreview(null);
      return;
    }
    if (arrowStart === square) {
      // Right-click without drag → toggle red highlight on this square.
      setHighlights((prev) =>
        prev.includes(square)
          ? prev.filter((s) => s !== square)
          : [...prev, square]
      );
    } else {
      // Right-drag → toggle arrow from arrowStart to square.
      setArrows((prev) => {
        const exists = prev.some(
          (arrow) => arrow.from === arrowStart && arrow.to === square
        );
        if (exists) {
          return prev.filter(
            (arrow) => !(arrow.from === arrowStart && arrow.to === square)
          );
        }
        return [...prev, { from: arrowStart, to: square }];
      });
    }
    setArrowStart(null);
    setArrowPreview(null);
  };

  const handleArrowHover = (event, square) => {
    if (!arrowStart || (event.buttons & 2) === 0) {
      return;
    }
    if (arrowStart === square) {
      setArrowPreview(null);
      return;
    }
    setArrowPreview({ from: arrowStart, to: square });
  };

  const resetToStart = () => {
    const nextGame = new Chess();
    gameRef.current = nextGame;
    const updated = nextGame.fen();
    setGameFen(updated);
    setFenInput(updated);
    setFenError("");
    setActiveSquare(null);
    setLegalTargets([]);
    clearAnnotations();
  };

  const undoMove = () => {
    const undone = gameRef.current.undo();
    if (!undone) return;
    const updated = gameRef.current.fen();
    setGameFen(updated);
    setFenInput(updated);
    setFenError("");
    setActiveSquare(null);
    setLegalTargets([]);
    clearAnnotations();
  };

  const handleRun = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    setChatError("");

    const trimmedFen = fenInput.trim();
    if (trimmedFen && trimmedFen !== gameRef.current.fen()) {
      const ok = syncFen(trimmedFen);
      if (!ok) {
        setError("Invalid FEN");
        setLoading(false);
        return;
      }
    }

    try {
      const response = await fetch("http://localhost:8001/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fen: gameRef.current.fen(),
          moves: null,
          prompt,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Request failed");
      }

      const data = await response.json();
      setResult(data);
      const freshChat = [];
      const trimmedPrompt = prompt.trim();
      if (trimmedPrompt) {
        freshChat.push({ role: "user", content: trimmedPrompt });
      }
      if (data.llm_answer) {
        freshChat.push({ role: "assistant", content: data.llm_answer });
      }
      setChatMessages(freshChat);
      setFollowUp("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFollowUp = async () => {
    if (!result) {
      return;
    }
    const question = followUp.trim();
    if (!question) {
      return;
    }
    setChatLoading(true);
    setChatError("");

    try {
      const response = await fetch("http://localhost:8001/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fen: result.fen || gameRef.current.fen(),
          prompt,
          question,
          history: chatMessages,
          engine_lines: result.engine_lines || [],
          candidate_lines: result.candidate_lines || [],
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Request failed");
      }

      const data = await response.json();
      setChatMessages((prev) => [
        ...prev,
        { role: "user", content: question },
        { role: "assistant", content: data.answer || "" },
      ]);
      setFollowUp("");
      if (data.error) {
        setChatError(data.error);
      }
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Chess Teacher</p>
          <h1>Explain the best move with engine-backed clarity.</h1>
          <p className="subhead">
            Feed a position, ask a question, and get a visual breakdown of engine
            lines, candidate moves, and a concise explanation.
          </p>
        </div>
        <div className="hero-card">
          <div className="label">Active Question</div>
          <div className="hero-prompt">{prompt || "Your prompt"}</div>
          <div className="hero-meta">
            <span>{gameFen ? "Custom FEN" : "Start position"}</span>
            {evalData?.stockfish && (
              <span>SF {formatSf(evalData.stockfish.score)}</span>
            )}
            {evalData?.alphazero && (
              <span>AZ {formatAz(evalData.alphazero.value_white)}</span>
            )}
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="panel-header">
            <h2>Position</h2>
            <p>Paste a FEN or play on the board, then run analysis.</p>
          </div>

          <div className="board-and-eval">
          <div className="eval-bars">
            <div
              className="eval-bar"
              title={`Stockfish (White POV): ${formatSf(evalData?.stockfish?.score)}`}
            >
              <div
                className="eval-fill"
                style={{ height: `${stockfishWhitePct(evalData?.stockfish?.score)}%` }}
              />
              <span className="eval-tag">SF</span>
              <span className="eval-num">{formatSf(evalData?.stockfish?.score)}</span>
            </div>
            <div
              className="eval-bar"
              title={`AlphaZero value (White POV): ${formatAz(evalData?.alphazero?.value_white)}`}
            >
              <div
                className="eval-fill"
                style={{ height: `${alphazeroWhitePct(evalData?.alphazero?.value_white)}%` }}
              />
              <span className="eval-tag">AZ</span>
              <span className="eval-num">{formatAz(evalData?.alphazero?.value_white)}</span>
            </div>
          </div>
          <div
            className="board-wrapper"
            onContextMenu={(event) => event.preventDefault()}
          >
            <svg className="arrow-layer" viewBox="0 0 100 100">
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="4"
                  markerHeight="4"
                  refX="3.2"
                  refY="2"
                  orient="auto"
                >
                  <polygon points="0 0, 4 2, 0 4" fill="#2b211b" />
                </marker>
              </defs>
              {arrows.map((arrow) => {
                const start = squareCenter(arrow.from);
                const end = squareCenter(arrow.to);
                return (
                  <line
                    key={`${arrow.from}-${arrow.to}`}
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    stroke="#2b211b"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    markerEnd="url(#arrowhead)"
                  />
                );
              })}
              {arrowPreview && (
                <line
                  x1={squareCenter(arrowPreview.from).x}
                  y1={squareCenter(arrowPreview.from).y}
                  x2={squareCenter(arrowPreview.to).x}
                  y2={squareCenter(arrowPreview.to).y}
                  stroke="#2b211b"
                  strokeWidth="1.2"
                  strokeLinecap="round"
                  strokeOpacity="0.45"
                  markerEnd="url(#arrowhead)"
                />
              )}
            </svg>
            <div className="board">
              {board.map((row, rIdx) =>
                row.map((piece, cIdx) => {
                  const square = `${FILES[cIdx]}${RANKS[rIdx]}`;
                  const light = (rIdx + cIdx) % 2 === 0;
                  const isActive = square === activeSquare;
                  const isTarget = legalTargets.includes(square);
                  const isHighlighted = highlights.includes(square);
                  const salMap =
                    showSaliency && saliency
                      ? saliencyMode === "value"
                        ? saliency.value_saliency
                        : saliency.move_saliency
                      : null;
                  const salWeight = salMap ? salMap[square] || 0 : 0;
                  const pieceSrc = piece ? PIECE_IMAGES[piece] : null;
                  const pieceObj = piece ? gameRef.current.get(square) : null;
                  const isDraggable =
                    pieceObj && pieceObj.color === gameRef.current.turn();

                  return (
                    <div
                      key={`${rIdx}-${cIdx}`}
                      className={`square ${light ? "light" : "dark"} ${
                        isActive ? "active" : ""
                      } ${isTarget ? "target" : ""} ${
                        isHighlighted ? "highlight" : ""
                      }`}
                      onClick={() => handleSquareClick(square)}
                      onDragOver={(event) => event.preventDefault()}
                      onDrop={(event) => handleDrop(event, square)}
                      onMouseDown={(event) => handleSquareMouseDown(event, square)}
                      onMouseUp={(event) => handleArrowEnd(event, square)}
                      onMouseEnter={(event) => handleArrowHover(event, square)}
                    >
                      {salWeight > 0.05 && (
                        <div
                          className="saliency-overlay"
                          style={{
                            backgroundColor: `rgba(226,59,46,${(0.72 * salWeight).toFixed(3)})`,
                          }}
                        />
                      )}
                      {cIdx === 0 && (
                        <span className="coord coord-rank">{RANKS[rIdx]}</span>
                      )}
                      {rIdx === 7 && (
                        <span className="coord coord-file">{FILES[cIdx]}</span>
                      )}
                      {pieceSrc && (
                        <img
                          src={pieceSrc}
                          alt=""
                          className="piece-img"
                          draggable={isDraggable}
                          onDragStart={(event) =>
                            handleDragStart(event, square)
                          }
                          onDragEnd={() => setDragFrom(null)}
                        />
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
          </div>

          <div className="saliency-controls">
            <label className="toggle">
              <input
                type="checkbox"
                checked={showSaliency}
                onChange={(event) => setShowSaliency(event.target.checked)}
              />
              <span>
                Show attention heatmap
                {saliencyLoading ? " (loading…)" : ""}
                {showSaliency && saliency && saliency.chosen_move
                  ? ` — move ${saliency.chosen_move}`
                  : ""}
              </span>
            </label>
            <span
              className="info-hint"
              tabIndex={0}
              role="note"
              aria-label="What the attention heatmap, policy/value, and SF vs AZ mean"
            >
              <span className="info-icon">i</span>
              <span className="info-tip" role="tooltip">
                <strong>Attention heatmap</strong>
                <p>
                  Highlights the squares the neural net&rsquo;s self-attention
                  weighted most while judging this position. Two views:
                </p>
                <ul>
                  <li>
                    <b>Value saliency</b> &mdash; squares that most drove the
                    position <i>evaluation</i> (the value head).
                  </li>
                  <li>
                    <b>Move saliency</b> &mdash; squares that most justified the
                    net&rsquo;s <i>chosen move</i> (the policy head).
                  </li>
                </ul>
                <strong>Policy vs value</strong>
                <p>
                  <b>Policy</b> is the net&rsquo;s move preference &mdash; a
                  probability spread over legal moves (&ldquo;what to
                  play&rdquo;). <b>Value</b> is one score for the whole position
                  (&ldquo;who&rsquo;s winning&rdquo;, &minus;1&hellip;+1 from the
                  side to move). Attention explains <i>both</i>: which squares fed
                  each answer.
                </p>
                <strong>Stockfish (SF) vs AlphaZero (AZ)</strong>
                <p>
                  <b>SF</b> is the classical search engine &mdash; extremely
                  strong, reports centipawns/mate. <b>AZ</b> is this project&rsquo;s
                  trained attention network &mdash; it outputs policy + value and,
                  unlike SF, can <i>explain itself</i> through this heatmap. The
                  two eval bars let you compare their verdicts.
                </p>
              </span>
            </span>
            {showSaliency && (
              <select
                value={saliencyMode}
                onChange={(event) => setSaliencyMode(event.target.value)}
              >
                <option value="value">Value saliency (what drives the eval)</option>
                <option value="move">Move saliency (what justifies the move)</option>
              </select>
            )}
          </div>

          <div className="form">
            <label>
              FEN
              <textarea
                rows={3}
                value={fenInput}
                onChange={(event) => setFenInput(event.target.value)}
                placeholder="Paste a FEN string"
              />
            </label>
            <button
              type="button"
              className="ghost"
              onClick={() => syncFen(fenInput)}
            >
              Apply FEN to board
            </button>
            <div className="button-row">
              <button
                type="button"
                className="ghost"
                onClick={undoMove}
                disabled={gameRef.current.history().length === 0}
                title="Take back the last move played on the board"
              >
                ↩ Undo move
              </button>
              <button type="button" className="ghost" onClick={resetToStart}>
                Reset to starting position
              </button>
            </div>
            {fenError && <div className="error">{fenError}</div>}
            <label>
              Prompt
              <input
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Why cant I play Nf3 here?"
              />
            </label>

            <button type="button" onClick={handleRun} disabled={loading}>
              {loading ? "Analyzing..." : "Run analysis"}
            </button>
            {error && <div className="error">{error}</div>}
          </div>
        </section>

        <section className="panel results">
          <div className="panel-header">
            <h2>Results</h2>
            <p>Engine lines, candidate ideas, and the LLM explanation.</p>
          </div>

          {!result && !loading && (
            <div className="empty">Run analysis to populate insights.</div>
          )}

          {result && (
            <div className="result-grid">
              <div className="result-card">
                <h3>Engine lines</h3>
                <ul>
                  {result.engine_lines?.map((line, idx) => (
                    <li key={`engine-${idx}`}>
                      <span className="tag">{line.formatted_score}</span>
                      <span>{line.moves.join(" ")}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="result-card">
                <h3>Candidate lines</h3>
                <ul>
                  {result.candidate_lines?.length ? (
                    result.candidate_lines.map((line, idx) => (
                      <li key={`candidate-${idx}`}>
                        <div className="candidate-line">
                          <strong>{line.label}</strong>
                          <span>{formatMoveLine(line)}</span>
                        </div>
                        {line.tags?.length ? (
                          <div className="tags">
                            {line.tags.map((tag, tagIdx) => (
                              <span className="chip" key={`${idx}-${tagIdx}`}>
                                {tag.move}: {tag.tags.join(", ")}
                              </span>
                            ))}
                          </div>
                        ) : null}
                      </li>
                    ))
                  ) : (
                    <li className="muted">No candidate lines returned.</li>
                  )}
                </ul>
              </div>

              <div className="result-card wide">
                <h3>LLM explanation</h3>
                <pre className="explanation">
                  {result.llm_answer || "No explanation yet."}
                </pre>
                {result.llm_error && (
                  <p className="muted">{result.llm_error}</p>
                )}
              </div>

              <div className="result-card wide chat">
                <h3>Coach chat</h3>
                <div className="chat-messages">
                  {chatMessages.length ? (
                    chatMessages.map((message, idx) => (
                      <div
                        key={`chat-${idx}`}
                        className={`chat-message ${message.role}`}
                      >
                        {message.content}
                      </div>
                    ))
                  ) : (
                    <div className="muted">Ask a follow-up question to begin.</div>
                  )}
                </div>
                <div className="chat-input-row">
                  <input
                    value={followUp}
                    onChange={(event) => setFollowUp(event.target.value)}
                    placeholder="Ask a follow-up question"
                  />
                  <button
                    type="button"
                    onClick={handleFollowUp}
                    disabled={!result || chatLoading || !followUp.trim()}
                  >
                    {chatLoading ? "Sending..." : "Send"}
                  </button>
                </div>
                {chatError && <p className="muted">{chatError}</p>}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
