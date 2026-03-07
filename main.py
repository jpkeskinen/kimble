#!/usr/bin/env python3
"""Kimble-pelin käynnistys."""
from board import NUM_PLAYERS, PLAYER_COLORS
from player import Player
from game import Game


def ask_player_type(player_id: int) -> bool:
    """Kysy onko pelaaja ihminen (True) vai tietokone (False)."""
    color = PLAYER_COLORS[player_id]
    while True:
        choice = input(f"  Pelaaja {player_id + 1} ({color}) – (i)hminen vai (t)ietokone? ").strip().lower()
        if choice in ("i", "ihminen", "h", "human"):
            return True
        if choice in ("t", "tietokone", "c", "computer", "a", "ai"):
            return False
        print("  Kirjoita 'i' tai 't'.")


def main():
    print("=" * 50)
    print("        KIMBLE")
    print("=" * 50)
    print("Valitse pelaajatyypit (4 pelaajaa):\n")

    players = []
    for i in range(NUM_PLAYERS):
        is_human = ask_player_type(i)
        players.append(Player(i, is_human))

    print("\nPeli alkaa!\n")
    game = Game(players)
    game.run()


if __name__ == "__main__":
    main()
