"""PUCT Monte-Carlo Tree Search over the attention policy/value net.

Edge-based statistics with alternating-sign backup. Values are always in the
perspective of the side to move at the node. NN evaluations are cached by FEN.
Supports policy-only play (``n_sims<=0``) for a fast, strong baseline.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from encoding import encode_position, legal_moves_with_indices


class Node:
    __slots__ = ("to_move", "expanded", "moves", "P", "N", "W", "Q", "children", "n_total")

    def __init__(self, to_move: bool):
        self.to_move = to_move
        self.expanded = False
        self.moves: List[chess.Move] = []
        self.P: np.ndarray = np.zeros(0, dtype=np.float32)
        self.N: np.ndarray = np.zeros(0, dtype=np.float32)
        self.W: np.ndarray = np.zeros(0, dtype=np.float32)
        self.Q: np.ndarray = np.zeros(0, dtype=np.float32)
        self.children: Dict[int, "Node"] = {}
        self.n_total: int = 0


class MCTS:
    def __init__(
        self,
        net,
        device,
        c_puct: float = 1.5,
        n_sims: int = 200,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.0,
        fpu: float = 0.2,
        batch_size: int = 16,
        vloss: float = 1.0,
    ):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.fpu = fpu
        self.batch_size = max(1, int(batch_size))
        self.vloss = vloss
        self._cache: Dict[str, Tuple[List[chess.Move], np.ndarray, float]] = {}
        self.net.eval()

    # -- NN evaluation (priors over legal moves + value), cached by FEN --------
    def evaluate(self, board: chess.Board) -> Tuple[List[chess.Move], np.ndarray, float]:
        key = board._transposition_key() if hasattr(board, "_transposition_key") else board.fen()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        piece_ids, g, mirrored = encode_position(board)
        moves, idxs = legal_moves_with_indices(board, mirrored)
        pid_t = torch.from_numpy(piece_ids).unsqueeze(0).to(self.device)
        g_t = torch.from_numpy(g).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value, _ = self.net(pid_t, g_t)
        logits = logits[0].detach().float().cpu().numpy()
        if len(moves) == 0:
            priors = np.zeros(0, dtype=np.float32)
        else:
            sel = logits[idxs]
            sel = sel - sel.max()
            priors = np.exp(sel)
            priors /= priors.sum()
        result = (moves, priors.astype(np.float32), float(value.item()))
        self._cache[key] = result
        return result

    def _evaluate_batch(self, boards: List[chess.Board]):
        """One NN forward for many boards. Returns list of (moves, priors, value).
        Uses the FEN cache for hits."""
        results: List = [None] * len(boards)
        pend_boards, pend_idx, pend_keys = [], [], []
        for i, b in enumerate(boards):
            key = b._transposition_key() if hasattr(b, "_transposition_key") else b.fen()
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                pend_boards.append(b)
                pend_idx.append(i)
                pend_keys.append(key)
        if pend_boards:
            pid_list, g_list, meta = [], [], []
            for b in pend_boards:
                pid, g, mir = encode_position(b)
                moves, idxs = legal_moves_with_indices(b, mir)
                pid_list.append(pid)
                g_list.append(g)
                meta.append((moves, idxs))
            pid_t = torch.from_numpy(np.stack(pid_list)).to(self.device)
            g_t = torch.from_numpy(np.stack(g_list)).to(self.device)
            with torch.no_grad():
                logits, value, _ = self.net(pid_t, g_t)
            logits = logits.detach().float().cpu().numpy()
            value = value.detach().float().cpu().numpy()
            for j, (moves, idxs) in enumerate(meta):
                if len(moves) == 0:
                    priors = np.zeros(0, dtype=np.float32)
                else:
                    sel = logits[j][idxs]
                    sel = sel - sel.max()
                    priors = np.exp(sel)
                    priors /= priors.sum()
                res = (moves, priors.astype(np.float32), float(value[j]))
                self._cache[pend_keys[j]] = res
                results[pend_idx[j]] = res
        return results

    @staticmethod
    def _set_node(node: Node, moves, priors) -> None:
        node.moves = moves
        node.P = priors
        node.N = np.zeros(len(moves), dtype=np.float32)
        node.W = np.zeros(len(moves), dtype=np.float32)
        node.Q = np.zeros(len(moves), dtype=np.float32)
        node.expanded = True

    def _expand(self, node: Node, board: chess.Board) -> float:
        moves, priors, value = self.evaluate(board)
        self._set_node(node, moves, priors)
        return value

    def _apply_vloss(self, node: Node, a: int) -> None:
        node.N[a] += 1
        node.W[a] -= self.vloss
        node.Q[a] = node.W[a] / node.N[a]
        node.n_total += 1

    def _revert(self, path: List[Tuple[Node, int]]) -> None:
        for node, a in path:
            node.N[a] -= 1
            node.W[a] += self.vloss
            node.n_total -= 1
            node.Q[a] = node.W[a] / node.N[a] if node.N[a] > 0 else 0.0

    def _backup_vloss(self, path: List[Tuple[Node, int]], leaf_value: float) -> None:
        """Remove the virtual loss applied during descent and add the real value."""
        value = -leaf_value
        for node, a in reversed(path):
            # undo virtual loss
            node.N[a] -= 1
            node.W[a] += self.vloss
            node.n_total -= 1
            # apply real visit
            node.N[a] += 1
            node.W[a] += value
            node.n_total += 1
            node.Q[a] = node.W[a] / node.N[a]
            value = -value

    def _select(self, node: Node) -> int:
        # PUCT with first-play-urgency for unvisited edges
        sqrt_total = math.sqrt(max(node.n_total, 1))
        q = np.where(node.N > 0, node.Q, self.fpu)
        u = self.c_puct * node.P * sqrt_total / (1.0 + node.N)
        return int(np.argmax(q + u))

    def _add_root_noise(self, node: Node) -> None:
        if self.dirichlet_eps <= 0 or len(node.moves) == 0:
            return
        noise = np.random.default_rng().dirichlet([self.dirichlet_alpha] * len(node.moves))
        node.P = (1 - self.dirichlet_eps) * node.P + self.dirichlet_eps * noise.astype(np.float32)

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        # value from perspective of side to move at this terminal node
        result = board.result(claim_draw=True)
        if result == "1/2-1/2":
            return 0.0
        # side to move is checkmated (loss) if it's game over with a decisive result
        return -1.0

    def search(self, board: chess.Board) -> Node:
        """Batched PUCT search with virtual loss (batch_size leaves per NN forward)."""
        root = Node(board.turn)
        moves, priors, _ = self._evaluate_batch([board])[0]
        self._set_node(root, moves, priors)
        self._add_root_noise(root)

        sims = max(self.n_sims, 1)
        done = 0
        while done < sims:
            batch = min(self.batch_size, sims - done)
            collected: List[Tuple[Node, chess.Board, List[Tuple[Node, int]]]] = []
            seen = set()
            tries = 0
            while len(collected) < batch and tries < batch * 4:
                tries += 1
                node = root
                b = board.copy()
                path: List[Tuple[Node, int]] = []
                while node.expanded and not b.is_game_over() and len(node.moves) > 0:
                    a = self._select(node)
                    path.append((node, a))
                    self._apply_vloss(node, a)
                    b.push(node.moves[a])
                    child = node.children.get(a)
                    if child is None:
                        child = Node(b.turn)
                        node.children[a] = child
                    node = child
                    if not node.expanded:
                        break
                if b.is_game_over():
                    self._backup_vloss(path, self._terminal_value(b))
                    done += 1
                    continue
                if node.expanded:
                    # dead-end (expanded, no moves) - shouldn't happen; revert
                    self._revert(path)
                    continue
                if id(node) in seen:
                    # same leaf already queued this batch: revert and flush what we have
                    self._revert(path)
                    break
                seen.add(id(node))
                collected.append((node, b, path))

            if not collected:
                if done < sims:  # progress guard: one sequential sim
                    done += self._one_sequential_sim(root, board)
                continue

            results = self._evaluate_batch([b for _, b, _ in collected])
            for (node, _b, path), (mvs, prs, val) in zip(collected, results):
                self._set_node(node, mvs, prs)
                self._backup_vloss(path, val)
                done += 1
        return root

    def _one_sequential_sim(self, root: Node, board: chess.Board) -> int:
        node = root
        b = board.copy()
        path: List[Tuple[Node, int]] = []
        while node.expanded and not b.is_game_over() and len(node.moves) > 0:
            a = self._select(node)
            path.append((node, a))
            b.push(node.moves[a])
            child = node.children.get(a)
            if child is None:
                child = Node(b.turn)
                node.children[a] = child
            node = child
            if not node.expanded:
                break
        if b.is_game_over():
            leaf_value = self._terminal_value(b)
        else:
            leaf_value = self._expand(node, b)
        value = -leaf_value
        for n, a in reversed(path):
            n.N[a] += 1
            n.W[a] += value
            n.Q[a] = n.W[a] / n.N[a]
            n.n_total += 1
            value = -value
        return 1

    def policy_only_move(self, board: chess.Board) -> chess.Move:
        moves, priors, _ = self.evaluate(board)
        return moves[int(np.argmax(priors))]

    def select_move(
        self, board: chess.Board, temperature: float = 0.0
    ) -> Tuple[chess.Move, Optional[Node]]:
        if self.n_sims <= 0:
            return self.policy_only_move(board), None
        root = self.search(board)
        visits = root.N
        if temperature <= 1e-6:
            a = int(np.argmax(visits))
        else:
            dist = visits ** (1.0 / temperature)
            dist = dist / dist.sum()
            a = int(np.random.default_rng().choice(len(visits), p=dist))
        return root.moves[a], root

    def clear_cache(self) -> None:
        self._cache.clear()
