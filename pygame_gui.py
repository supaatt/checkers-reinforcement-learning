"""
pygame GUI
interface to play against the AI.
  - Click-to-move with move highlighting
  - AI thinking indicator
  - Undo, new game, difficulty controls
"""

import pygame
import sys
import numpy as np
import threading
import time

from config import (
    GUIConfig as G, BOARD_SIZE, EMPTY,
    BLACK_MAN, BLACK_KING, WHITE_MAN, WHITE_KING,
    BLACK, WHITE, DEVICE,
)
from checkers_env import CheckersState
from neural_network import NetworkWrapper
from mcts_fast import MCTS


class CheckersGUI:
    def __init__(self, model_path=None, ai_simulations=100):
        pygame.init()
        pygame.display.set_caption("fuckass checkers please work please please")

        self.screen = pygame.display.set_mode((G.WINDOW_W, G.WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Helvetica", G.FONT_SIZE)
        self.title_font = pygame.font.SysFont("Helvetica", G.TITLE_FONT_SIZE, bold=True)
        self.small_font = pygame.font.SysFont("Helvetica", 14)

        # Gamestate
        self.state = CheckersState()
        self.selected_piece = None
        self.valid_moves_for_selected = []
        self.last_move = None
        self.game_over = False
        self.winner = 0
        self.history = []

        # AI
        self.ai_simulations = ai_simulations
        self.human_color = WHITE
        self.ai_thinking = False
        self.ai_move_result = None

        self.nnet = NetworkWrapper()
        if model_path:
            try:
                self.nnet.load(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Using untrained network.")
        else:
            print("No model path provided. Using untrained network.")
        self.mcts = MCTS(self.nnet)

        self.message = "Your turn (White)"
        self.message_timer = 0

        self.difficulties = {
            'Easy': 25, 'Medium': 100, 'Hard': 200, 'Harder': 400, 'LooiLe' :900
        }
        self.current_difficulty = 'Medium'

    def run(self):
        running = True
        if self.state.current_player != self.human_color and not self.game_over:
            self._start_ai_turn()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            if self.ai_move_result is not None:
                self._apply_ai_move()

            self._draw()
            self.clock.tick(G.FPS)

        pygame.quit()
        sys.exit()

    #Input

    def _handle_click(self, pos):
        x, y = pos
        if x < G.BOARD_PX and y < G.WINDOW_H:
            if self.game_over or self.ai_thinking:
                return
            if self.state.current_player != self.human_color:
                return
            col = x // G.SQUARE_SIZE
            row = y // G.SQUARE_SIZE
            self._handle_board_click(row, col)
        else:
            self._handle_panel_click(x, y)

    def _handle_board_click(self, row, col):
        if self.selected_piece is not None:
            for move in self.valid_moves_for_selected:
                final_r, final_c = move[-1][2], move[-1][3]
                if (final_r, final_c) == (row, col):
                    self._make_human_move(move)
                    return

        if self.state.is_player_piece(row, col, self.human_color):
            self.selected_piece = (row, col)
            self.valid_moves_for_selected = [
                m for m in self.state.get_legal_moves()
                if m[0][0] == row and m[0][1] == col
            ]
        else:
            self.selected_piece = None
            self.valid_moves_for_selected = []

    def _handle_panel_click(self, x, y):
        panel_x = G.BOARD_PX + 10
        btn_w = G.INFO_PANEL_W - 20

        if panel_x <= x <= panel_x + btn_w and 400 <= y <= 440:
            self._new_game()
        elif panel_x <= x <= panel_x + btn_w and 450 <= y <= 490:
            self._undo()
        elif 540 <= y <= 580:
            diffs = list(self.difficulties.keys())
            btn_each = btn_w // len(diffs)
            idx = (x - panel_x) // btn_each
            if 0 <= idx < len(diffs):
                self.current_difficulty = diffs[idx]
                self.ai_simulations = self.difficulties[self.current_difficulty]

    def _handle_key(self, key):
        if key == pygame.K_n:
            self._new_game()
        elif key == pygame.K_u:
            self._undo()
        elif key == pygame.K_q:
            pygame.quit()
            sys.exit()

    #Game Actions

    def _make_human_move(self, move):
        self.history.append(self.state.copy())
        from_rc = (move[0][0], move[0][1])
        to_rc = (move[-1][2], move[-1][3])
        self.state = self.state.apply_move(move)
        self.last_move = (from_rc, to_rc)
        self.selected_piece = None
        self.valid_moves_for_selected = []

        done, winner = self.state.is_terminal()
        if done:
            self.game_over = True
            self.winner = winner
            self._set_message(self._winner_text())
        else:
            self._start_ai_turn()

    def _start_ai_turn(self):
        self.ai_thinking = True
        self.ai_move_result = None
        self._set_message("AI is thinking...")

        def ai_worker():
            try:
                _, move, _ = self.mcts.get_action(
                    self.state, temperature=0.1,
                    num_simulations=self.ai_simulations, add_noise = False
                )
                self.ai_move_result = move
            except Exception as e:
                print(f"AI error: {e}")
                moves = self.state.get_legal_moves()
                if moves:
                    self.ai_move_result = moves[0]

        thread = threading.Thread(target=ai_worker, daemon=True)
        thread.start()

    def _apply_ai_move(self):
        move = self.ai_move_result
        self.ai_move_result = None
        self.ai_thinking = False

        if move is None:
            self.game_over = True
            self.winner = self.human_color
            self._set_message("AI has no moves. You win!")
            return

        self.history.append(self.state.copy())
        from_rc = (move[0][0], move[0][1])
        to_rc = (move[-1][2], move[-1][3])
        self.state = self.state.apply_move(move)
        self.last_move = (from_rc, to_rc)

        done, winner = self.state.is_terminal()
        if done:
            self.game_over = True
            self.winner = winner
            self._set_message(self._winner_text())
        else:
            self._set_message("Your turn (White)")

    def _new_game(self):
        self.state = CheckersState()
        self.selected_piece = None
        self.valid_moves_for_selected = []
        self.last_move = None
        self.game_over = False
        self.winner = 0
        self.history = []
        self.ai_thinking = False
        self.ai_move_result = None
        self._set_message("New game! Your turn (White)")
        if self.state.current_player != self.human_color:
            self._start_ai_turn()

    def _undo(self):
        if len(self.history) >= 2 and not self.ai_thinking:
            self.state = self.history.pop()
            self.state = self.history.pop()
            self.game_over = False
            self.selected_piece = None
            self.valid_moves_for_selected = []
            self.last_move = None
            self._set_message("Undone. Your turn (White)")
        elif len(self.history) == 1 and not self.ai_thinking:
            self.state = self.history.pop()
            self.game_over = False
            self.selected_piece = None
            self.valid_moves_for_selected = []
            self.last_move = None

    def _set_message(self, msg):
        self.message = msg
        self.message_timer = time.time()

    def _winner_text(self):
        if self.winner == 0:
            return "Game Over - Draw!"
        elif self.winner == self.human_color:
            return "Game Over - You Win!"
        else:
            return "Game Over - AI Wins!"

    #Drawing

    def _draw(self):
        self.screen.fill(G.BG_COLOR)
        self._draw_board()
        self._draw_pieces()
        self._draw_highlights()
        self._draw_info_panel()
        pygame.display.flip()

    def _draw_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                rect = pygame.Rect(
                    c * G.SQUARE_SIZE, r * G.SQUARE_SIZE,
                    G.SQUARE_SIZE, G.SQUARE_SIZE,
                )
                color = G.DARK_SQUARE if (r + c) % 2 == 1 else G.LIGHT_SQUARE
                pygame.draw.rect(self.screen, color, rect)

                if c == 0:
                    label = self.small_font.render(str(r), True, (150, 150, 150))
                    self.screen.blit(label, (3, r * G.SQUARE_SIZE + 2))
                if r == BOARD_SIZE - 1:
                    label = self.small_font.render(str(c), True, (150, 150, 150))
                    self.screen.blit(label, (c * G.SQUARE_SIZE + G.SQUARE_SIZE - 12,
                                             G.WINDOW_H - 16))

    def _draw_pieces(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.state.board[r][c]
                if piece == EMPTY:
                    continue

                cx = c * G.SQUARE_SIZE + G.SQUARE_SIZE // 2
                cy = r * G.SQUARE_SIZE + G.SQUARE_SIZE // 2

                if piece in (BLACK_MAN, BLACK_KING):
                    color = G.BLACK_PIECE
                    border = G.BLACK_BORDER
                else:
                    color = G.WHITE_PIECE
                    border = G.WHITE_BORDER

                pygame.draw.circle(self.screen, (0, 0, 0),
                                   (cx + 3, cy + 3), G.PIECE_RADIUS)
                pygame.draw.circle(self.screen, color, (cx, cy), G.PIECE_RADIUS)
                pygame.draw.circle(self.screen, border, (cx, cy),
                                   G.PIECE_RADIUS, G.PIECE_BORDER_W)
                pygame.draw.circle(self.screen, border, (cx, cy),
                                   G.PIECE_RADIUS - 10, 1)

                if piece in (BLACK_KING, WHITE_KING):
                    self._draw_crown(cx, cy)

    def _draw_crown(self, cx, cy):
        color = G.KING_CROWN
        s = G.PIECE_RADIUS // 2
        points = [
            (cx - s, cy + s // 3),
            (cx - s, cy - s // 3),
            (cx - s // 2, cy),
            (cx, cy - s // 2),
            (cx + s // 2, cy),
            (cx + s, cy - s // 3),
            (cx + s, cy + s // 3),
        ]
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (200, 170, 0), points, 2)

    def _draw_highlights(self):
        if self.last_move:
            for rc in self.last_move:
                r, c = rc
                surf = pygame.Surface((G.SQUARE_SIZE, G.SQUARE_SIZE), pygame.SRCALPHA)
                surf.fill((*G.LAST_MOVE_FROM, 80))
                self.screen.blit(surf, (c * G.SQUARE_SIZE, r * G.SQUARE_SIZE))

        if self.selected_piece:
            r, c = self.selected_piece
            surf = pygame.Surface((G.SQUARE_SIZE, G.SQUARE_SIZE), pygame.SRCALPHA)
            surf.fill((*G.HIGHLIGHT_COLOR, 100))
            self.screen.blit(surf, (c * G.SQUARE_SIZE, r * G.SQUARE_SIZE))

            for move in self.valid_moves_for_selected:
                dr, dc = move[-1][2], move[-1][3]
                cx = dc * G.SQUARE_SIZE + G.SQUARE_SIZE // 2
                cy = dr * G.SQUARE_SIZE + G.SQUARE_SIZE // 2
                surf = pygame.Surface((G.SQUARE_SIZE, G.SQUARE_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*G.VALID_MOVE_COLOR, 150),
                                   (G.SQUARE_SIZE // 2, G.SQUARE_SIZE // 2), 15)
                self.screen.blit(surf, (dc * G.SQUARE_SIZE, dr * G.SQUARE_SIZE))

    def _draw_info_panel(self):
        panel_rect = pygame.Rect(G.BOARD_PX, 0, G.INFO_PANEL_W, G.WINDOW_H)
        pygame.draw.rect(self.screen, (40, 40, 45), panel_rect)
        pygame.draw.line(self.screen, (60, 60, 65),
                         (G.BOARD_PX, 0), (G.BOARD_PX, G.WINDOW_H), 2)

        x = G.BOARD_PX + 15
        y = 15

        title = self.title_font.render("LAM (LooiLeAttMurt)", True, (220, 220, 220))
        self.screen.blit(title, (x, y))
        y += 35
        subtitle = self.font.render("Checkers", True, (180, 180, 180))
        self.screen.blit(subtitle, (x, y))
        y += 40

        dev_text = self.small_font.render(f"Device: {DEVICE}", True, (120, 200, 120))
        self.screen.blit(dev_text, (x, y))
        y += 25
        sim_text = self.small_font.render(
            f"MCTS Sims: {self.ai_simulations}", True, (120, 180, 200))
        self.screen.blit(sim_text, (x, y))
        y += 35

        b_men = np.sum(self.state.board == BLACK_MAN)
        b_kings = np.sum(self.state.board == BLACK_KING)
        w_men = np.sum(self.state.board == WHITE_MAN)
        w_kings = np.sum(self.state.board == WHITE_KING)

        pygame.draw.rect(self.screen, (55, 55, 60),
                         pygame.Rect(x - 5, y - 5, G.INFO_PANEL_W - 20, 80),
                         border_radius=8)
        ai_label = self.font.render("AI (Black)", True, (200, 200, 200))
        self.screen.blit(ai_label, (x, y))
        ai_count = self.small_font.render(f"  Men: {b_men}  Kings: {b_kings}",
                                           True, (170, 170, 170))
        self.screen.blit(ai_count, (x, y + 22))

        y += 45
        you_label = self.font.render("You (White)", True, (200, 200, 200))
        self.screen.blit(you_label, (x, y))
        you_count = self.small_font.render(f"  Men: {w_men}  Kings: {w_kings}",
                                            True, (170, 170, 170))
        self.screen.blit(you_count, (x, y + 22))
        y += 55

        move_text = self.small_font.render(
            f"Move: {self.state.move_count}", True, (150, 150, 150))
        self.screen.blit(move_text, (x, y))
        y += 30

        if self.ai_thinking:
            dots = "." * (int(time.time() * 2) % 4)
            msg = self.font.render(f"AI thinking{dots}", True, (255, 165, 0))
        elif self.game_over:
            if self.winner == 0:
                msg_color = (255, 255, 100)
            elif self.winner == self.human_color:
                msg_color = (100, 255, 100)
            else:
                msg_color = (255, 100, 100)
            msg = self.font.render(self.message, True, msg_color)
        else:
            msg = self.font.render(self.message, True, (200, 200, 200))
        self.screen.blit(msg, (x, y))
        y += 40

        btn_w = G.INFO_PANEL_W - 30
        self._draw_button("New Game (N)", x, 400, btn_w, 35, (70, 130, 70))
        self._draw_button("Undo (U)", x, 450, btn_w, 35, (130, 100, 70))

        y_diff = 510
        diff_label = self.font.render("Difficulty:", True, (180, 180, 180))
        self.screen.blit(diff_label, (x, y_diff))
        y_diff += 28

        diffs = list(self.difficulties.keys())
        btn_each = btn_w // len(diffs)
        for i, d in enumerate(diffs):
            color = (80, 140, 200) if d == self.current_difficulty else (60, 60, 65)
            self._draw_button(d, x + i * btn_each, y_diff, btn_each - 2, 30,
                              color, font=self.small_font)

        y_help = G.WINDOW_H - 80
        controls = ["Controls:", "  Click piece, then destination",
                     "  N = New Game  |  U = Undo", "  Q = Quit"]
        for line in controls:
            help_text = self.small_font.render(line, True, (100, 100, 100))
            self.screen.blit(help_text, (x, y_help))
            y_help += 16

    def _draw_button(self, text, x, y, w, h, color, font=None):
        if font is None:
            font = self.font
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        pygame.draw.rect(self.screen, (100, 100, 100), rect, 1, border_radius=6)
        label = font.render(text, True, (220, 220, 220))
        lx = x + (w - label.get_width()) // 2
        ly = y + (h - label.get_height()) // 2
        self.screen.blit(label, (lx, ly))
