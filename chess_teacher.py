import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
try:
    from huggingface_hub import InferenceClient
except ImportError:  # pragma: no cover - optional dependency
    InferenceClient = None

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

CENTER_SQUARES = {
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
}

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TOKEN_FILE_NAME = "api_token.txt"
DEFAULT_ROUTER_TIMEOUT = 60


@dataclass
class LineResult:
    label: str
    moves: List[str]
    score: Dict[str, int]
    tags: List[Dict[str, object]]
    source: str


def find_engine_path(cli_path: Optional[str]) -> Optional[Path]:
    if cli_path:
        path = Path(cli_path)
        return path if path.exists() else None

    base = Path(__file__).resolve().parent
    candidates = [
        base / "stockfish" / "stockfish-windows-x86-64-avx2.exe",
        base / "stockfish-windows-x86-64-avx2.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_hf_token() -> Optional[str]:
    token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token

    token_path = Path(__file__).resolve().parent / TOKEN_FILE_NAME
    if not token_path.exists():
        return None

    content = token_path.read_text(encoding="utf-8").strip()
    return content or None


def parse_move(board: chess.Board, token: str) -> chess.Move:
    token = token.strip()
    if not token:
        raise ValueError("empty move")

    try:
        move = chess.Move.from_uci(token.lower())
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    try:
        return board.parse_san(token)
    except ValueError:
        pass

    if token[0] in "nbrqk" and token[0].islower():
        try:
            return board.parse_san(token[0].upper() + token[1:])
        except ValueError:
            pass

    raise ValueError(f"illegal or unknown move: {token}")


def score_to_dict(score: chess.engine.PovScore) -> Dict[str, int]:
    if score.is_mate():
        return {"mate": score.mate()}
    return {"cp": score.score(mate_score=100000)}


def format_score(score: Dict[str, int]) -> str:
    if "mate" in score and score["mate"] is not None:
        return f"mate {score['mate']}"
    return f"{score.get('cp', 0) / 100:.2f}"


def pv_to_san(board: chess.Board, moves: List[chess.Move]) -> List[str]:
    b = board.copy()
    san_moves = []
    for move in moves:
        san_moves.append(b.san(move))
        b.push(move)
    return san_moves


def pinned_squares(board: chess.Board, color: chess.Color) -> List[int]:
    squares = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and board.is_pinned(color, square):
            squares.append(square)
    return squares


def creates_fork(board: chess.Board, square: int, mover_color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if not piece:
        return False

    targets = []
    for target_sq in board.attacks(square):
        target = board.piece_at(target_sq)
        if target and target.color != mover_color:
            targets.append(target)

    if not targets:
        return False

    high_value = [
        t for t in targets if PIECE_VALUES[t.piece_type] >= 3 or t.piece_type == chess.KING
    ]
    return len(high_value) >= 2


def is_hanging(board: chess.Board, square: int, color: chess.Color) -> bool:
    piece = board.piece_at(square)
    if not piece:
        return False
    attackers = board.attackers(not color, square)
    defenders = board.attackers(color, square)
    return len(attackers) > 0 and len(defenders) == 0


def move_tags(board: chess.Board, move: chess.Move) -> List[str]:
    tags = []
    piece = board.piece_at(move.from_square)
    if not piece:
        return tags

    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            tags.append(f"captures {captured.symbol().upper()}")
        else:
            tags.append("captures")

    if board.is_castling(move):
        tags.append("castle")
    if board.gives_check(move):
        tags.append("check")
    if move.promotion:
        tags.append("promotion")

    if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        if chess.square_rank(move.from_square) in (0, 7):
            tags.append("develops minor")

    if move.to_square in CENTER_SQUARES:
        tags.append("occupies center")

    opponent = not board.turn
    before_pins = set(pinned_squares(board, opponent))

    tmp = board.copy()
    tmp.push(move)

    after_pins = set(pinned_squares(tmp, opponent))
    if len(after_pins) > len(before_pins):
        tags.append("creates pin")

    if creates_fork(tmp, move.to_square, piece.color):
        tags.append("fork threat")

    if is_hanging(tmp, move.to_square, piece.color):
        tags.append("piece may be hanging")

    return tags


def line_tags(board: chess.Board, moves: List[str]) -> List[Dict[str, object]]:
    tags = []
    b = board.copy()
    for ply_index, san in enumerate(moves, start=1):
        try:
            move = parse_move(b, san)
        except ValueError:
            break
        entry = {"ply": ply_index, "move": b.san(move), "tags": move_tags(b, move)}
        tags.append(entry)
        b.push(move)
    return tags


def ply_label(board: chess.Board, ply_index: int, san: str) -> str:
    move_number = board.fullmove_number + (ply_index - 1) // 2
    is_white_move = (ply_index % 2 == 1) == (board.turn == chess.WHITE)
    if is_white_move:
        return f"{move_number}.{san}"
    return f"{move_number}...{san}"


def format_tags_for_prompt(board: chess.Board, tags: List[Dict[str, object]]) -> str:
    parts = []
    for entry in tags:
        if not entry["tags"]:
            continue
        label = ply_label(board, entry["ply"], entry["move"])
        parts.append(f"{label}: {', '.join(entry['tags'])}")
    return "; ".join(parts)


def engine_top_lines(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    top: int,
    pv_plies: int,
) -> List[LineResult]:
    info_list = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=top)
    if isinstance(info_list, dict):
        info_list = [info_list]

    lines = []
    for idx, info in enumerate(info_list, start=1):
        pv = info.get("pv", [])[:pv_plies]
        san_moves = pv_to_san(board, pv)
        score = score_to_dict(info["score"].pov(board.turn))
        tags = line_tags(board, san_moves)
        lines.append(
            LineResult(
                label=f"engine_{idx}",
                moves=san_moves,
                score=score,
                tags=tags,
                source="engine",
            )
        )
    return lines


def find_prompt_moves(prompt: str, board: chess.Board) -> List[str]:
    lower_prompt = prompt.lower()
    moves = []
    for move in board.legal_moves:
        san = board.san(move)
        uci = move.uci()
        if re.search(rf"\b{re.escape(san.lower())}\b", lower_prompt):
            moves.append(san)
            continue
        if re.search(rf"\b{re.escape(uci)}\b", lower_prompt):
            moves.append(san)
    return moves


def safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def hf_generate(
    prompt: str,
    model: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    if InferenceClient is None:
        raise RuntimeError("huggingface_hub is not installed. Run 'pip install huggingface_hub'.")
    token = load_hf_token()
    if not token:
        raise RuntimeError("Missing HUGGINGFACE_API_TOKEN, HF_TOKEN, or api_token.txt")

    client_error = None
    try:
        try:
            client = InferenceClient(api_key=token)
        except TypeError:
            client = InferenceClient(token=token)

        if hasattr(client, "chat") and hasattr(client.chat, "completions"):
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            message = completion.choices[0].message
            text = getattr(message, "content", None)
            if text is None and isinstance(message, dict):
                text = message.get("content")
            if text:
                return str(text).strip()
        else:
            completion = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            choice = completion.choices[0]
            message = getattr(choice, "message", None) or {}
            text = getattr(message, "content", None)
            if text is None and isinstance(message, dict):
                text = message.get("content")
            if text:
                return str(text).strip()
    except Exception as exc:
        client_error = exc

    url = "https://router.huggingface.co/v1/chat/completions"
    timeout_seconds = int(os.getenv("HF_ROUTER_TIMEOUT", str(DEFAULT_ROUTER_TIMEOUT)))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"HF router error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error contacting Hugging Face router: {exc}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Hugging Face router request timed out") from exc
    except Exception as exc:
        raise RuntimeError(f"Hugging Face router error: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Unexpected Hugging Face router response") from exc

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF router error: {data['error']}")

    choices = data.get("choices", []) if isinstance(data, dict) else []
    if not choices:
        if client_error:
            raise RuntimeError(f"HF client failed: {client_error}")
        raise RuntimeError("Unexpected Hugging Face router response")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    text = message.get("content") if isinstance(message, dict) else None
    if not text:
        raise RuntimeError("Unexpected Hugging Face router response")

    return str(text).strip()


def llm_candidate_lines(
    board: chess.Board,
    prompt: str,
    max_candidates: int,
    model: str,
) -> List[Tuple[str, List[str]]]:
    legal_san = [board.san(m) for m in board.legal_moves]
    legal_list = ", ".join(legal_san)

    llm_prompt = (
        "You are a chess assistant. Return only valid JSON.\n"
        "Provide up to {max_candidates} candidate move lines that directly address the user question.\n"
        "Use SAN notation only, and only moves from the provided legal list.\n"
        "Output schema: {{\"candidates\": [{{\"label\": \"...\", \"moves\": [\"Nf3\", \"Nc6\"]}}]}}.\n\n"
        "Position FEN: {fen}\n"
        "Side to move: {side}\n"
        "Legal moves (SAN): {legal}\n"
        "User question: {question}\n"
    ).format(
        max_candidates=max_candidates,
        fen=board.fen(),
        side="White" if board.turn == chess.WHITE else "Black",
        legal=legal_list,
        question=prompt,
    )

    try:
        raw = hf_generate(llm_prompt, model=model, max_new_tokens=192, temperature=0.3)
    except RuntimeError as exc:
        print(f"LLM candidate generation skipped: {exc}")
        return []
    data = safe_json_loads(raw)
    if not data or "candidates" not in data:
        return []

    results = []
    for entry in data.get("candidates", [])[:max_candidates]:
        label = str(entry.get("label", "candidate"))
        moves = entry.get("moves", [])
        if not isinstance(moves, list) or not moves:
            continue
        results.append((label, [str(m) for m in moves]))

    return results


def fallback_explain(
    board: chess.Board,
    prompt: str,
    engine_lines: List[LineResult],
    candidate_lines: List[LineResult],
    error: str,
) -> str:
    parts = [
        "LLM unavailable; showing engine-only guidance.",
        f"Reason: {error}",
        f"Prompt: {prompt}",
    ]

    if engine_lines:
        best = engine_lines[0]
        parts.append(
            "Best engine line: "
            f"{format_score(best.score)} | {' '.join(best.moves)}"
        )

    if candidate_lines:
        parts.append("Candidate line notes:")
        for line in candidate_lines:
            tags = format_tags_for_prompt(board, line.tags)
            line_text = f"{line.label}: {format_score(line.score)} | {' '.join(line.moves)}"
            if tags:
                line_text += f" | tags: {tags}"
            parts.append(line_text)

    return "\n".join(parts)


def extend_line_with_engine(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    pv_plies: int,
    moves: List[str],
) -> List[str]:
    b = board.copy()
    san_moves = []
    for move_str in moves:
        try:
            move = parse_move(b, move_str)
        except ValueError:
            break
        san_moves.append(b.san(move))
        b.push(move)
        if len(san_moves) >= pv_plies:
            return san_moves

    remaining = pv_plies - len(san_moves)
    if remaining <= 0:
        return san_moves

    info = engine.analyse(b, chess.engine.Limit(depth=depth))
    pv = info.get("pv", [])[:remaining]
    san_moves.extend(pv_to_san(b, pv))
    return san_moves


def evaluate_line(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    moves: List[str],
) -> Dict[str, int]:
    b = board.copy()
    for move_str in moves:
        try:
            move = parse_move(b, move_str)
        except ValueError:
            break
        b.push(move)
    info = engine.analyse(b, chess.engine.Limit(depth=depth))
    return score_to_dict(info["score"].pov(board.turn))


def build_candidate_lines(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    prompt: str,
    depth: int,
    pv_plies: int,
    max_candidates: int,
    model: str,
    enable_llm: bool,
) -> List[LineResult]:
    candidates = []
    seen = set()

    prompt_moves = find_prompt_moves(prompt, board)
    for move in prompt_moves:
        if (move,) not in seen:
            candidates.append(("prompt", [move]))
            seen.add((move,))

    if enable_llm:
        for label, moves in llm_candidate_lines(board, prompt, max_candidates, model):
            key = tuple(moves)
            if key in seen:
                continue
            candidates.append((label, moves))
            seen.add(key)

    results = []
    for label, moves in candidates[:max_candidates]:
        line_moves = extend_line_with_engine(board, engine, depth, pv_plies, moves)
        score = evaluate_line(board, engine, depth, line_moves)
        tags = line_tags(board, line_moves)
        results.append(
            LineResult(
                label=label,
                moves=line_moves,
                score=score,
                tags=tags,
                source="candidate",
            )
        )

    return results


def format_line_for_prompt(board: chess.Board, line: LineResult) -> str:
    moves = " ".join(line.moves)
    tags = format_tags_for_prompt(board, line.tags)
    return f"{line.label}: {format_score(line.score)} | {moves} | tags: {tags}"


def response_needs_rewrite(text: str) -> bool:
    if not text:
        return True
    sample = text.strip()
    if len(sample) < 80:
        return True
    lowered = sample.lower()
    if "candidate line" in lowered or "engine line" in lowered or "tags:" in lowered:
        return True
    list_lines = sum(
        1
        for line in sample.splitlines()
        if re.match(r"^\s*(\d+[\).]|[-*])\s+", line)
    )
    return list_lines >= 2


def basic_explain(
    board: chess.Board,
    prompt: str,
    engine_lines: List[LineResult],
    candidate_lines: List[LineResult],
) -> str:
    if not engine_lines:
        return "No engine lines available to explain the position."

    best = engine_lines[0]
    parts = []
    if best.moves:
        parts.append(f"Best move is {best.moves[0]} ({format_score(best.score)}).")
    else:
        parts.append(f"Best line score is {format_score(best.score)}.")

    if best.tags:
        first_tags = best.tags[0].get("tags", [])
        if first_tags:
            parts.append(f"Key idea: {', '.join(first_tags[:3])}.")

    if len(best.moves) > 1:
        line = " ".join(best.moves[:4])
        parts.append(f"Sample line: {line}.")

    prompt_moves = find_prompt_moves(prompt, board)
    if prompt_moves:
        parts.append(
            f"About {prompt_moves[0]}: it scores worse than the best line and"
            " allows counterplay."
        )

    return " ".join(parts)


def llm_explain(
    board: chess.Board,
    prompt: str,
    engine_lines: List[LineResult],
    candidate_lines: List[LineResult],
    model: str,
) -> str:
    engine_text = "\n".join(
        f"{idx + 1}) {format_score(line.score)}: {' '.join(line.moves)}"
        for idx, line in enumerate(engine_lines)
    )

    candidate_text = "\n".join(format_line_for_prompt(board, line) for line in candidate_lines)

    expl_prompt = (
        "You are a chess coach. Answer with explanation only.\n"
        "Explain why the best move is best and why the questioned moves fail if applicable.\n"
        "Avoid listing full move sequences or tags; mention at most one short line (up to 4 plies).\n"
        "Keep the answer focused and practical in 2-4 short paragraphs.\n\n"
        "Position FEN: {fen}\n"
        "Side to move: {side}\n"
        "User question: {question}\n\n"
        "Engine top lines (score from side to move, positive is better):\n{engine}\n\n"
        "Candidate lines with tags:\n{candidates}\n"
    ).format(
        fen=board.fen(),
        side="White" if board.turn == chess.WHITE else "Black",
        question=prompt,
        engine=engine_text or "(none)",
        candidates=candidate_text or "(none)",
    )

    response = hf_generate(expl_prompt, model=model, max_new_tokens=256, temperature=0.3)
    if response_needs_rewrite(response):
        rewrite_prompt = (
            "Rewrite the answer into a concise explanation.\n"
            "Do not list or enumerate move lines or tags.\n"
            "Use 2-4 sentences. Mention at most one key line (up to 4 plies).\n"
            "No headings.\n\n"
            "Position FEN: {fen}\n"
            "Side to move: {side}\n"
            "User question: {question}\n\n"
            "Engine top lines:\n{engine}\n\n"
            "Candidate lines with tags:\n{candidates}\n"
        ).format(
            fen=board.fen(),
            side="White" if board.turn == chess.WHITE else "Black",
            question=prompt,
            engine=engine_text or "(none)",
            candidates=candidate_text or "(none)",
        )
        response = hf_generate(
            rewrite_prompt,
            model=model,
            max_new_tokens=192,
            temperature=0.2,
        )
        if response_needs_rewrite(response):
            return basic_explain(board, prompt, engine_lines, candidate_lines)

    return response


def build_board(fen: Optional[str], moves: Optional[List[str]]) -> chess.Board:
    board = chess.Board(fen) if fen else chess.Board()
    if moves:
        for token in moves:
            move = parse_move(board, token)
            board.push(move)
    return board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-assisted Stockfish explainer")
    parser.add_argument("--fen", help="FEN string for the position")
    parser.add_argument("--moves", nargs="*", help="Moves from start (SAN or UCI)")
    parser.add_argument("--prompt", help="Question for the coach")
    parser.add_argument("--stockfish", help="Path to Stockfish engine")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--pv-plies", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--llm", choices=["hf", "none"], default="hf")
    parser.add_argument("--model", default=os.getenv("HF_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.prompt:
        args.prompt = input("Prompt: ").strip()

    board = build_board(args.fen, args.moves)

    engine_path = find_engine_path(args.stockfish)
    if not engine_path:
        print("Could not find Stockfish engine. Use --stockfish to set the path.")
        return 2

    engine_lines = []
    candidate_lines = []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": args.threads})
        engine_lines = engine_top_lines(board, engine, args.depth, args.top, args.pv_plies)
        candidate_lines = build_candidate_lines(
            board,
            engine,
            args.prompt,
            args.depth,
            args.pv_plies,
            args.max_candidates,
            args.model,
            enable_llm=args.llm == "hf",
        )

    print(f"Position FEN: {board.fen()}")
    print(f"Prompt: {args.prompt}")
    print()

    print(f"Top engine lines (depth {args.depth}, {args.pv_plies} plies):")
    for idx, line in enumerate(engine_lines, start=1):
        print(f"{idx}) {format_score(line.score)}: {' '.join(line.moves)}")

    if candidate_lines:
        print()
        print("Candidate lines:")
        for line in candidate_lines:
            print(format_line_for_prompt(board, line))

    if args.llm == "none":
        return 0

    try:
        answer = llm_explain(board, args.prompt, engine_lines, candidate_lines, args.model)
    except RuntimeError as exc:
        answer = fallback_explain(board, args.prompt, engine_lines, candidate_lines, str(exc))

    print()
    print("LLM answer:")
    print(answer)

    if args.debug:
        payload = {
            "engine_lines": [line.__dict__ for line in engine_lines],
            "candidate_lines": [line.__dict__ for line in candidate_lines],
        }
        print()
        print("Debug:")
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
