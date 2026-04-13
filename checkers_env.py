"""
AlphaZero Checkers — Full Game Engine
======================================
Standard American/English checkers (8x8):
  - Black moves first
  - Men move diagonally forward; kings move diagonally forward & backward
  - Mandatory jumps (longest capture sequence enforced)
  - Promotion on back rank
  - Draw after 80 moves with no captures/promotions
"""

import numpy as np
from copy import deepcopy
from config import (
    BOARD_SIZE, NUM_SQUARES, EMPTY,
    BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING,
    BLACK, WHITE, NetworkConfig,
)

# ── Coordinate helpers ──────────────────────────────────────────────

def sq_to_rc(sq):
    """Convert square index (0-31) to (row, col) on 8x8 board."""
    row = sq // 4
    col = (sq % 4) * 2 + (1 - row % 2)
    return row, col

def rc_to_sq(row, col):
    """Convert (row, col) to square index. Returns -1 if not a dark square."""
    if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
        return -1
    if (row + col) % 2 == 0:
        return -1
    return row * 4 + (col // 2)

SQ_TO_RC = {sq: sq_to_rc(sq) for sq in range(NUM_SQUARES)}
RC_TO_SQ = {}
for r in range(BOARD_SIZE):
    for c in range(BOARD_SIZE):
        s = rc_to_sq(r, c)
        if s >= 0:
            RC_TO_SQ[(r, c)] = s


class CheckersState:
    """
    Immutable-style game state for checkers.
    Board stored as numpy array of shape (8, 8).
    """

    def __init__(self, board=None, current_player=BLACK, no_progress_count=0,
                 move_count=0):
        if board is None:
            self.board = self._initial_board()
        else:
            self.board = board.copy()
        self.current_player = current_player
        self.no_progress_count = no_progress_count
        self.move_count = move_count

    @staticmethod
    def _initial_board():
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for sq in range(NUM_SQUARES):
            r, c = SQ_TO_RC[sq]
            if r < 3:
                board[r][c] = BLACK_MAN
            elif r > 4:
                board[r][c] = WHITE_MAN
        return board

    def copy(self):
        return CheckersState(
            board=self.board.copy(),
            current_player=self.current_player,
            no_progress_count=self.no_progress_count,
            move_count=self.move_count,
        )

    # ── Piece queries ───────────────────────────────────────────────

    def is_player_piece(self, r, c, player):
        p = self.board[r][c]
        if player == BLACK:
            return p in (BLACK_MAN, BLACK_KING)
        return p in (WHITE_MAN, WHITE_KING)

    def is_king(self, r, c):
        return self.board[r][c] in (BLACK_KING, WHITE_KING)

    def is_opponent_piece(self, r, c, player):
        return self.is_player_piece(r, c, -player)

    # ── Move generation ─────────────────────────────────────────────

    def _forward_dirs(self, player, r, c):
        piece = self.board[r][c]
        dirs = []
        if piece in (BLACK_MAN, BLACK_KING, WHITE_KING):
            dirs += [(1, -1), (1, 1)]
        if piece in (WHITE_MAN, WHITE_KING, BLACK_KING):
            dirs += [(-1, -1), (-1, 1)]
        return dirs

    def _get_jumps(self, r, c, player, visited=None):
        if visited is None:
            visited = set()
        dirs = self._forward_dirs(player, r, c)
        sequences = []

        for dr, dc in dirs:
            mr, mc = r + dr, c + dc
            lr, lc = r + 2 * dr, c + 2 * dc
            if (0 <= lr < BOARD_SIZE and 0 <= lc < BOARD_SIZE
                    and self.is_opponent_piece(mr, mc, player)
                    and self.board[lr][lc] == EMPTY
                    and (mr, mc) not in visited):
                visited_new = visited | {(mr, mc)}
                old_piece = self.board[r][c]
                old_mid = self.board[mr][mc]

                promoted = False
                piece_after = old_piece
                if player == BLACK and lr == BOARD_SIZE - 1 and old_piece == BLACK_MAN:
                    piece_after = BLACK_KING
                    promoted = True
                elif player == WHITE and lr == 0 and old_piece == WHITE_MAN:
                    piece_after = WHITE_KING
                    promoted = True

                self.board[lr][lc] = piece_after
                self.board[r][c] = EMPTY
                self.board[mr][mc] = EMPTY

                if promoted:
                    continuations = []
                else:
                    continuations = self._get_jumps(lr, lc, player, visited_new)

                self.board[r][c] = old_piece
                self.board[mr][mc] = old_mid
                self.board[lr][lc] = EMPTY

                step = (r, c, lr, lc)
                if continuations:
                    for cont in continuations:
                        sequences.append([step] + cont)
                else:
                    sequences.append([step])

        return sequences

    def get_legal_moves(self):
        player = self.current_player
        jump_moves = []
        simple_moves = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if not self.is_player_piece(r, c, player):
                    continue

                jumps = self._get_jumps(r, c, player)
                jump_moves.extend(jumps)

                for dr, dc in self._forward_dirs(player, r, c):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE
                            and self.board[nr][nc] == EMPTY):
                        simple_moves.append([(r, c, nr, nc)])

        if jump_moves:
            max_len = max(len(j) for j in jump_moves)
            return [j for j in jump_moves if len(j) == max_len]
        return simple_moves

    # ── Apply move ──────────────────────────────────────────────────

    def apply_move(self, move):
        new_state = self.copy()
        captured = False

        for step in move:
            sr, sc, dr, dc = step
            piece = new_state.board[sr][sc]
            new_state.board[sr][sc] = EMPTY

            if abs(dr - sr) == 2:
                mr, mc = (sr + dr) // 2, (sc + dc) // 2
                new_state.board[mr][mc] = EMPTY
                captured = True

            if piece == BLACK_MAN and dr == BOARD_SIZE - 1:
                piece = BLACK_KING
            elif piece == WHITE_MAN and dr == 0:
                piece = WHITE_KING

            new_state.board[dr][dc] = piece

        new_state.move_count += 1
        if captured:
            new_state.no_progress_count = 0
        else:
            new_state.no_progress_count += 1

        new_state.current_player = -self.current_player
        return new_state

    # ── Terminal checks ─────────────────────────────────────────────

    def is_terminal(self):
        if self.no_progress_count >= 80:
            return True, 0

        if self.move_count >= 200:
            return True, 0

        moves = self.get_legal_moves()
        if not moves:
            return True, -self.current_player

        has_black = np.any(np.isin(self.board, [BLACK_MAN, BLACK_KING]))
        has_white = np.any(np.isin(self.board, [WHITE_MAN, WHITE_KING]))
        if not has_black:
            return True, WHITE
        if not has_white:
            return True, BLACK

        return False, 0

    # ── State encoding for neural network ───────────────────────────

    def encode(self):
        planes = np.zeros((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        if self.current_player == BLACK:
            my_man, my_king = BLACK_MAN, BLACK_KING
            opp_man, opp_king = WHITE_MAN, WHITE_KING
        else:
            my_man, my_king = WHITE_MAN, WHITE_KING
            opp_man, opp_king = BLACK_MAN, BLACK_KING

        planes[0] = (self.board == my_man).astype(np.float32)
        planes[1] = (self.board == my_king).astype(np.float32)
        planes[2] = (self.board == opp_man).astype(np.float32)
        planes[3] = (self.board == opp_king).astype(np.float32)
        if self.current_player == BLACK:
            planes[4] = 1.0

        return planes

    # ── Move encoding / decoding ────────────────────────────────────

    @staticmethod
    def move_to_index(move):
        sr, sc = move[0][0], move[0][1]
        dr, dc = move[-1][2], move[-1][3]
        src_sq = rc_to_sq(sr, sc)
        dst_sq = rc_to_sq(dr, dc)
        if src_sq < 0 or dst_sq < 0:
            raise ValueError(f"Invalid move squares: ({sr},{sc})->({dr},{dc})")
        return src_sq * 32 + dst_sq

    @staticmethod
    def index_to_src_dst(index):
        return index // 32, index % 32

    def get_legal_move_mask(self):
        mask = np.zeros(NetworkConfig.POLICY_SIZE, dtype=np.float32)
        for move in self.get_legal_moves():
            idx = self.move_to_index(move)
            mask[idx] = 1.0
        return mask

    def get_move_from_index(self, index, legal_moves=None):
        if legal_moves is None:
            legal_moves = self.get_legal_moves()
        for move in legal_moves:
            if self.move_to_index(move) == index:
                return move
        return None

    def __repr__(self):
        symbols = {EMPTY: '.', BLACK_MAN: 'b', BLACK_KING: 'B',
                   WHITE_MAN: 'w', WHITE_KING: 'W'}
        lines = []
        for r in range(BOARD_SIZE):
            row_str = " ".join(symbols.get(self.board[r][c], '?')
                               for c in range(BOARD_SIZE))
            lines.append(f"  {r} {row_str}")
        header = "    " + " ".join(str(c) for c in range(BOARD_SIZE))
        player = "BLACK" if self.current_player == BLACK else "WHITE"
        return f"{header}\n" + "\n".join(lines) + f"\n  Turn: {player} | Move: {self.move_count}"
