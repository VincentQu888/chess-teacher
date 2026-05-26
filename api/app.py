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
