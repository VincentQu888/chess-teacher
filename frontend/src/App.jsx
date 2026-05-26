import { useMemo, useState } from "react";

const DEFAULT_FEN =
  "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
const DEFAULT_PROMPT = "why cant I play Nf3 here";

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["8", "7", "6", "5", "4", "3", "2", "1"];

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

function formatMoveLine(line) {
  const score = line.formatted_score || "?";
  const moves = line.moves?.join(" ") || "";
  return `${score}  ${moves}`.trim();
}

export default function App() {
  const [fen, setFen] = useState(DEFAULT_FEN);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [moves, setMoves] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const board = useMemo(() => parseFen(fen), [fen]);

  const handleRun = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    const moveList = moves
      .split(/\s+/)
      .map((move) => move.trim())
      .filter(Boolean);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fen,
          moves: moveList.length ? moveList : null,
          prompt,
        }),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || "Request failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
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
            <span>{fen ? "Custom FEN" : "Start position"}</span>
            <span>{moves ? "With move list" : "No moves"}</span>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="panel-header">
            <h2>Position</h2>
            <p>Paste a FEN and optional moves, then run analysis.</p>
          </div>

          <div className="board">
            {board.map((row, rIdx) =>
              row.map((piece, cIdx) => {
                const light = (rIdx + cIdx) % 2 === 0;
                const isWhite = piece && piece === piece.toUpperCase();
                return (
                  <div
                    key={`${rIdx}-${cIdx}`}
                    className={`square ${light ? "light" : "dark"}`}
                  >
                    {cIdx === 0 && (
                      <span className="coord coord-rank">{RANKS[rIdx]}</span>
                    )}
                    {rIdx === 7 && (
                      <span className="coord coord-file">{FILES[cIdx]}</span>
                    )}
                    {piece && (
                      <span
                        className={`piece ${isWhite ? "white" : "black"}`}
                      >
                        {piece.toUpperCase()}
                      </span>
                    )}
                  </div>
                );
              })
            )}
          </div>

          <div className="form">
            <label>
              FEN
              <textarea
                rows={3}
                value={fen}
                onChange={(event) => setFen(event.target.value)}
              />
            </label>
            <label>
              Moves (SAN or UCI, space-separated)
              <input
                value={moves}
                onChange={(event) => setMoves(event.target.value)}
                placeholder="e4 e5 Nf3 Nc6"
              />
            </label>
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
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
