#!/usr/bin/env python3

#    Kimble-simulaattori jonka avulla voi simuloida Kimbleä.
#    Copyright (C) 2026  Jukka-Pekka Keskinen

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Kimble-pelin käynnistys."""
from tqdm import tqdm
from board import NUM_PLAYERS, PLAYER_COLORS
from player import Player, STRATEGY_NAMES, STRATEGY_KEYS
from game import Game

_STRATEGY_MENU = "\n".join(
    f"    [{i + 1}] {name}" for i, name in enumerate(STRATEGY_NAMES.values())
)


def ask_player_setup(player_id: int, allow_human: bool = True) -> Player:
    """Kysy pelaajan tyyppi. allow_human=False tarkoittaa vain tietokonestrategioita."""
    color = PLAYER_COLORS[player_id]
    print(f"\n  Pelaaja {player_id + 1} ({color}):")
    low = 0 if allow_human else 1
    if allow_human:
        print("    [0] Ihminen")
    print(_STRATEGY_MENU)
    while True:
        try:
            choice = int(input(f"  Valinta ({low}-{len(STRATEGY_KEYS)}): ").strip())
            if allow_human and choice == 0:
                return Player(player_id, is_human=True)
            if 1 <= choice <= len(STRATEGY_KEYS):
                return Player(player_id, is_human=False, strategy=STRATEGY_KEYS[choice - 1])
        except (ValueError, EOFError):
            pass
        print(f"  Kirjoita numero {low}–{len(STRATEGY_KEYS)}.")


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
    print("\nValitse strategia kullekin pelaajalle:")
    template_players = [ask_player_setup(i, allow_human=False) for i in range(NUM_PLAYERS)]
    strategies = [p.strategy for p in template_players]

    wins = [0] * NUM_PLAYERS
    for _ in tqdm(range(n), desc="Simuloidaan", unit="peli"):
        players = [Player(i, is_human=False, strategy=strategies[i]) for i in range(NUM_PLAYERS)]
        winner = Game(players, verbose=False).run()
        wins[winner] += 1

    print(f"\nTulokset ({n} peliä):")
    print("-" * 55)
    for i, w in enumerate(wins):
        strategy_name = STRATEGY_NAMES[strategies[i]]
        print(f"  {PLAYER_COLORS[i]:10s} ({strategy_name:35s}): {w:5d}  ({100 * w / n:.1f} %)")
    print("-" * 55)


def main():
    print("=" * 50)
    print("        KIMBLE")
    print("=" * 50)

    while True:
        mode = input("\n(p)eli vai (s)imulaatio? ").strip().lower()
        if mode in ("p", "peli", "g", "game"):
            break
        if mode in ("s", "simulaatio", "sim"):
            n = ask_num_simulations()
            run_simulation(n)
            return
        print("  Kirjoita 'p' tai 's'.")

    print("\nValitse pelaajatyypit:")
    players = [ask_player_setup(i) for i in range(NUM_PLAYERS)]

    print("\nPeli alkaa!\n")
    Game(players).run()


if __name__ == "__main__":
    main()
