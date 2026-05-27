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
        candidate_lines = ct.build_candidate_lines(
            board,
            engine,
            request.prompt,
            request.depth,
            request.pv_plies,
            request.max_candidates,
            model,
            enable_llm=request.llm,
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
