"""Pelaaja-luokat: ihminen ja tietokone."""
import random
from board import Piece, PLAYER_COLORS, PLAYER_SYMBOLS, NUM_PLAYERS


class Player:
    def __init__(self, player_id: int, is_human: bool):
        self.player_id = player_id
        self.is_human = is_human
        self.name = PLAYER_COLORS[player_id]
        self.symbol = PLAYER_SYMBOLS[player_id]
        self.pieces = [Piece(player_id, i) for i in range(4)]

    def pieces_at_home(self):
        return [p for p in self.pieces if p.is_home()]

    def pieces_on_track(self):
        return [p for p in self.pieces if p.is_on_track()]

    def pieces_in_goal(self):
        return [p for p in self.pieces if p.is_in_goal()]

    def has_won(self) -> bool:
        return len(self.pieces_in_goal()) == 4

    def choose_piece(self, movable: list[Piece], die: int) -> Piece | None:
        """Valitse siirrettävä nappula. Ihminen valitsee itse, tietokone satunnaisesti."""
        if not movable:
            return None
        if not self.is_human:
            return random.choice(movable)

        # Ihmispelaaja
        print(f"\n  Liikutettavissa olevat nappulat (heitto: {die}):")
        for i, piece in enumerate(movable):
            if piece.is_home():
                loc = "kotipesässä"
            elif piece.is_in_goal():
                loc = f"maalialueella ruutu {piece.goal_index() + 1}"
            else:
                loc = f"radalla ruutu {piece.pos}"
            print(f"  [{i + 1}] Nappula {piece.idx + 1} ({loc})")

        while True:
            try:
                choice = input(f"  Valitse nappula (1-{len(movable)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(movable):
                    return movable[idx]
            except (ValueError, EOFError):
                pass
            print("  Virheellinen valinta, yritä uudelleen.")
