#!/usr/bin/env python3
"""Kimble-pelin käynnistys."""
from tqdm import tqdm
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


def ask_num_simulations() -> int:
    while True:
        try:
            n = int(input("  Kuinka monta peliä simuloidaan? ").strip())
            if n > 0:
                return n
        except (ValueError, EOFError):
            pass
        print("  Syötä positiivinen kokonaisluku.")


def run_simulation(n: int):
    wins = [0] * NUM_PLAYERS
    for _ in tqdm(range(n), desc="Simuloidaan", unit="peli"):
        players = [Player(i, is_human=False) for i in range(NUM_PLAYERS)]
        winner = Game(players, verbose=False).run()
        wins[winner] += 1

    print(f"\nTulokset ({n} peliä):")
    print("-" * 35)
    for i, w in enumerate(wins):
        print(f"  {PLAYER_COLORS[i]:10s}: {w:5d} voittoa  ({100 * w / n:.1f} %)")
    print("-" * 35)


def main():
    print("=" * 50)
    print("        KIMBLE")
    print("=" * 50)

    while True:
        mode = input("(p)eli vai (s)imulaatio? ").strip().lower()
        if mode in ("p", "peli", "g", "game"):
            break
        if mode in ("s", "simulaatio", "sim"):
            n = ask_num_simulations()
            run_simulation(n)
            return
        print("  Kirjoita 'p' tai 's'.")

    print("Valitse pelaajatyypit (4 pelaajaa):\n")
    players = []
    for i in range(NUM_PLAYERS):
        is_human = ask_player_type(i)
        players.append(Player(i, is_human))

    print("\nPeli alkaa!\n")
    Game(players).run()


if __name__ == "__main__":
    main()
