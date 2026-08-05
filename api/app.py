from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chess_teacher as ct


class AnalyzeRequest(BaseModel):
    fen: Optional[str] = None
    moves: Optional[List[str]] = None
    prompt: str
    depth: int = 16
    top: int = 3
    pv_plies: int = 10
    threads: int = 4
    max_candidates: int = 3
    llm: bool = True
    llm_candidates: bool = False  # extra LLM call to brainstorm candidate moves (slow)
    attention: bool = True  # include the neural-net attention-weighted board state
    model: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    fen: Optional[str] = None
    prompt: str
    question: str
    history: List[ChatMessage] = []
    engine_lines: List[dict] = []
    candidate_lines: List[dict] = []
    attention: bool = True  # include the neural-net attention-weighted board state
    model: Optional[str] = None


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def line_to_dict(line: ct.LineResult) -> dict:
    return {
        "label": line.label,
        "moves": line.moves,
        "score": line.score,
        "formatted_score": ct.format_score(line.score),
        "tags": line.tags,
        "source": line.source,
    }


def dict_to_line(data: dict, source: str) -> ct.LineResult:
    return ct.LineResult(
        label=str(data.get("label", source)),
        moves=list(data.get("moves", [])),
        score=data.get("score", {"cp": 0}),
        tags=list(data.get("tags", [])),
        source=source,
    )


def message_to_dict(message: ChatMessage) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        board = ct.build_board(request.fen, request.moves)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    engine_path = ct.find_engine_path(None)
    if not engine_path:
        raise HTTPException(status_code=500, detail="Stockfish engine not found")

    model = request.model or ct.DEFAULT_MODEL
    engine_lines = []
    candidate_lines = []

    with ct.chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": request.threads})
        engine_lines = ct.engine_top_lines(
            board,
            engine,
            request.depth,
            request.top,
            request.pv_plies,
        )
        # Engine multipv + user-mentioned moves already supply candidate lines;
        # skip the extra (slow) LLM candidate-generation call. Opt in via `llm_candidates`.
        candidate_lines = ct.build_candidate_lines(
            board,
            engine,
            request.prompt,
            request.depth,
            request.pv_plies,
            request.max_candidates,
            model,
            enable_llm=request.llm_candidates,
        )

        llm_answer = None
        llm_error = None
        if request.llm:
            try:
                llm_answer = ct.llm_explain(
                    board,
                    request.prompt,
                    engine_lines,
                    candidate_lines,
                    model,
                    include_attention=request.attention,
                )
            except RuntimeError as exc:
                llm_error = str(exc)
                llm_answer = ct.fallback_explain(
                    board,
                    request.prompt,
                    engine_lines,
                    candidate_lines,
                    llm_error,
                )
        else:
            llm_answer = ct.fallback_explain(
                board,
                request.prompt,
                engine_lines,
                candidate_lines,
                "LLM disabled",
            )

    return {
        "fen": board.fen(),
        "prompt": request.prompt,
        "engine_lines": [line_to_dict(line) for line in engine_lines],
        "candidate_lines": [line_to_dict(line) for line in candidate_lines],
        "llm_answer": llm_answer,
        "llm_error": llm_error,
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    try:
        board = ct.build_board(request.fen, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = request.model or ct.DEFAULT_MODEL
    engine_lines = [dict_to_line(line, "engine") for line in request.engine_lines]
    candidate_lines = [dict_to_line(line, "candidate") for line in request.candidate_lines]
    history = [message_to_dict(msg) for msg in request.history]

    hypothetical_lines: list = []
    engine_path = ct.find_engine_path(None)
    if engine_path:
        try:
            with ct.chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
                engine.configure({"Threads": 4})
                # If the client didn't pass engine lines, compute them so the
                # deterministic answer paths (best move / verdict / why-move) work.
                if not engine_lines:
                    engine_lines = ct.engine_top_lines(board, engine, 14, 3, 10)
                hypothetical_lines = ct.build_hypothetical_lines(
                    board,
                    engine,
                    request.question,
                    engine_lines,
                    depth=14,
                    pv_plies=8,
                    max_lines=4,
                )
        except Exception:
            hypothetical_lines = []

    try:
        answer = ct.llm_followup(
            board,
            request.prompt,
            request.question,
            history,
            engine_lines,
            candidate_lines,
            model,
            hypothetical_lines=hypothetical_lines,
            include_attention=request.attention,
        )
        error = None
    except RuntimeError as exc:
        error = str(exc)
        answer = ct.fallback_explain(
            board,
            request.question,
            engine_lines,
            candidate_lines,
            error,
        )

    return {
        "answer": answer,
        "error": error,
        "hypothetical_lines": [line_to_dict(line) for line in hypothetical_lines],
    }


class SaliencyRequest(BaseModel):
    fen: Optional[str] = None
    move: Optional[str] = None


@app.post("/saliency")
def saliency(request: SaliencyRequest) -> dict:
    """Per-square attention saliency for the board heatmap overlay (neural bot).
    Returns 0..1-normalised weight maps for the value head and the chosen move."""
    try:
        board = ct.build_board(request.fen, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = ct.attention_report_json(board)
    if not data:
        raise HTTPException(status_code=503, detail="attention model unavailable")
    return {
        "value": data.get("value"),
        "chosen_move": data.get("chosen_move"),
        "top_moves": data.get("top_moves", []),
        "value_saliency": data.get("value_saliency_full", {}),
        "move_saliency": data.get("move_saliency_full", {}),
    }
