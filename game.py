"""Kimble-pelilogiikka."""
import random
from board import (
    Piece, TRACK_SIZE, HOME_SIZE, NUM_PLAYERS,
    PLAYER_STARTS, PLAYER_COLORS, PLAYER_SYMBOLS, home_entry
)
from player import Player


def roll_die() -> int:
    return random.randint(1, 6)


def _compute_new_pos(current_pos: int, steps: int, player_id: int) -> int | None:
    """
    Laske nappulan uusi sijainti. Palauttaa None jos siirto ylittää maalialueen.
    current_pos: pääraidan absoluuttinen indeksi (0-27)
    """
    entry = home_entry(player_id)
    pos = current_pos
    for step in range(1, steps + 1):
        if pos == entry:
            # Astutaan maalialueelle; loput askeleet kuluvat siellä
            remaining = steps - step
            if remaining < HOME_SIZE:
                return 100 + remaining
            return None  # yliammennus
        pos = (pos + 1) % TRACK_SIZE
    return pos


def get_piece_at(all_players: list[Player], track_pos: int) -> Piece | None:
    """Etsi nappula tietystä pääradan ruudusta."""
    for player in all_players:
        for piece in player.pieces:
            if piece.is_on_track() and piece.pos == track_pos:
                return piece
    return None


def get_eating_pieces(
    player: Player, movable: list[Piece], all_players: list[Player], die: int
) -> set[Piece]:
    """Palauta joukko nappuloista, jotka syövät vastustajan tällä siirrolla."""
    eating = set()
    for piece in movable:
        if piece.is_home():
            new_pos = PLAYER_STARTS[player.player_id]
        elif piece.is_on_track():
            new_pos = _compute_new_pos(piece.pos, die, player.player_id)
        else:
            continue  # maalialueella ei voi syödä
        if new_pos is not None and 0 <= new_pos < TRACK_SIZE:
            victim = get_piece_at(all_players, new_pos)
            if victim is not None and victim.player != player.player_id:
                eating.add(piece)
    return eating


def get_threatened_pieces(player: Player, all_players: list[Player]) -> set[Piece]:
    """Palauta joukko pelaajan radalla olevista nappuloista, joita vastustaja voi syödä seuraavalla siirrolla."""
    threatened = set()
    for piece in player.pieces:
        if not piece.is_on_track():
            continue
        for opponent in all_players:
            if opponent.player_id == player.player_id:
                continue
            for opp_piece in opponent.pieces:
                if not opp_piece.is_on_track():
                    continue
                for die in range(1, 7):
                    if _compute_new_pos(opp_piece.pos, die, opponent.player_id) == piece.pos:
                        threatened.add(piece)
                        break
    return threatened


def get_movable_pieces(player: Player, die: int, all_players: list[Player]) -> list[Piece]:
    """Palauta lista nappuloista, joita voidaan siirtää."""
    movable = []
    home_piece_added = False  # Kotipesästä lisätään korkeintaan yksi nappula

    for piece in player.pieces:
        if piece.is_home():
            if die == 6 and not home_piece_added:
                start = PLAYER_STARTS[player.player_id]
                occupant = get_piece_at(all_players, start)
                # Voi lähteä jos ruutu on tyhjä tai siellä on vastustaja (torkku)
                if occupant is None or occupant.player != player.player_id:
                    movable.append(piece)
                    home_piece_added = True
        elif piece.is_on_track():
            new_pos = _compute_new_pos(piece.pos, die, player.player_id)
            if new_pos is not None:
                # Ei voi siirtyä omalle nappulalle (pääradalla)
                if 0 <= new_pos < TRACK_SIZE:
                    occupant = get_piece_at(all_players, new_pos)
                    if occupant is not None and occupant.player == player.player_id:
                        continue  # blokki
                movable.append(piece)
        elif piece.is_in_goal():
            gi = piece.goal_index()
            new_gi = gi + die
            if new_gi < HOME_SIZE:
                # Tarkista ettei kohderuudussa ole jo oma nappula
                dest_occupied = any(
                    p is not piece and p.is_in_goal() and p.goal_index() == new_gi
                    for p in player.pieces
                )
                if not dest_occupied:
                    movable.append(piece)

    return movable


def move_piece(piece: Piece, die: int, player: Player, all_players: list[Player], verbose: bool = True) -> "Piece | None":
    """Siirrä nappulaa ja tee tarvittavat toimenpiteet (torkkaaminen). Palauttaa syödyn nappulan tai None."""
    if piece.is_home():
        new_pos = PLAYER_STARTS[player.player_id]
    elif piece.is_in_goal():
        new_pos = 100 + piece.goal_index() + die
    else:
        new_pos = _compute_new_pos(piece.pos, die, player.player_id)

    if new_pos is None:
        return None  # ei pitäisi tapahtua

    # Torkkaaminen: vastustajan nappula pääradalla palaa kotiin
    victim = None
    if 0 <= new_pos < TRACK_SIZE:
        victim = get_piece_at(all_players, new_pos)
        if victim is not None and victim.player != player.player_id:
            if verbose:
                print(f"  Torkku! {PLAYER_COLORS[victim.player]}'n nappula {victim.idx + 1} palaa kotipesään.")
            victim.pos = -1
        else:
            victim = None

    piece.pos = new_pos
    return victim


class Game:
    def __init__(self, players: list[Player], verbose: bool = True):
        self.players = players
        self.turn = 0
        self.verbose = verbose
        # Anna jokaiselle pelaajalle viittaus koko pelaajalistaan (mm. NN-strategiaa varten)
        for p in players:
            p._all_players = players

    def current_player(self) -> Player:
        return self.players[self.turn % NUM_PLAYERS]

    def run(self) -> int:
        """Pelaa peli loppuun ja palauta voittajan player_id."""
        if self.verbose:
            print_board(self.players)
        while True:
            player = self.current_player()
            self._play_turn(player)
            if player.has_won():
                if self.verbose:
                    print_board(self.players)
                    print(f"\n*** {player.name} voitti pelin! Onnittelut! ***\n")
                return player.player_id
            self.turn += 1

    def _play_turn(self, player: Player):
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Vuoro: {player.name} ({player.type_label})")

        extra_rolls = 0
        while True:
            die = roll_die()
            if self.verbose:
                print(f"  Noppa: {die}")

            movable = get_movable_pieces(player, die, self.players)

            if not movable:
                if self.verbose:
                    print("  Ei siirrettäviä nappuloita.")
                break

            needs_eat = player.uses_eating
            needs_threat = player.uses_defense
            can_eat = get_eating_pieces(player, movable, self.players, die) if needs_eat else set()
            threatened = get_threatened_pieces(player, self.players) if needs_threat else set()
            piece = player.choose_piece(movable, die, can_eat=can_eat, threatened=threatened)
            if piece is None:
                break

            was_home = piece.is_home()
            old_pos = piece.pos
            old_dist = player._piece_distance(piece)
            victim = move_piece(piece, die, player, self.players, verbose=self.verbose)
            new_dist = player._piece_distance(piece)
            player.on_piece_moved(piece, was_home)

            # Reward shaping harjoittelua varten
            if player._training_nn and player._trajectory:
                step_reward = player._pending_reward
                player._pending_reward = 0.0
                step_reward += (new_dist - old_dist) * 0.005  # etenemispalkkio
                if victim is not None:
                    step_reward += 0.3  # vastustajan syöminen
                if piece.is_in_goal() and new_dist == TRACK_SIZE + HOME_SIZE:
                    step_reward += 0.5  # nappula viimeiseen maalislottiin
                lp, _, v, ent = player._trajectory[-1]
                player._trajectory[-1] = (lp, step_reward, v, ent)

            # Syödyksi tulemisen rangaistus syödyn nappulan omistajalle
            if victim is not None:
                victim_player = self.players[victim.player]
                if victim_player._training_nn:
                    victim_player._pending_reward -= 0.2
            if self.verbose:
                _print_move(player, piece, old_pos, die)
                print_board(self.players)

            if player.has_won():
                break

            if die == 6:
                extra_rolls += 1
                if extra_rolls < 3:
                    if self.verbose:
                        print(f"  Kuutonen! {player.name} saa heittää uudelleen.")
                    continue
                else:
                    if self.verbose:
                        print("  Kolmas kuutonen peräkkäin – vuoro päättyy.")
            break


def _print_move(player: Player, piece: Piece, old_pos: int, die: int):
    if old_pos == -1:
        print(f"  Nappula {piece.idx + 1} lähti kotipesästä radalle (ruutu {piece.pos}).")
    elif piece.is_in_goal():
        if old_pos >= 100:
            print(f"  Nappula {piece.idx + 1} eteni maalialueella (ruutu {piece.goal_index() + 1}/4).")
        else:
            print(f"  Nappula {piece.idx + 1} siirtyi maalialueelle (ruutu {piece.goal_index() + 1}/4).")
    else:
        print(f"  Nappula {piece.idx + 1}: ruutu {old_pos} → {piece.pos}.")


# ---------------------------------------------------------------------------
# ASCII-näyttö
# ---------------------------------------------------------------------------

def _piece_str(all_players: list[Player], track_pos: int) -> str:
    pieces_here = []
    for player in all_players:
        for piece in player.pieces:
            if piece.is_on_track() and piece.pos == track_pos:
                pieces_here.append(piece)
    if not pieces_here:
        return ".."
    if len(pieces_here) == 1:
        p = pieces_here[0]
        return f"{PLAYER_SYMBOLS[p.player]}{p.idx + 1}"
    return f"{len(pieces_here)}x"


def print_board(all_players: list[Player]):
    print("\n" + "=" * 62)
    print("KIMBLE - PELILAUTA")
    print("=" * 62)

    row1 = []
    row2 = []
    for i in range(14):
        s = _piece_str(all_players, i)
        marker = "*" if i in PLAYER_STARTS else " "
        row1.append(f"{marker}{i:02d}[{s}]")
    for i in range(14, 28):
        s = _piece_str(all_players, i)
        marker = "*" if i in PLAYER_STARTS else " "
        row2.append(f"{marker}{i:02d}[{s}]")

    print("Rata (0-13) :")
    print(" ".join(row1))
    print("Rata (14-27):")
    print(" ".join(row2))
    print()

    for player in all_players:
        home_count = len(player.pieces_at_home())
        track_count = len(player.pieces_on_track())
        goal_slots = ""
        for i in range(HOME_SIZE):
            slot = next((p for p in player.pieces_in_goal() if p.goal_index() == i), None)
            goal_slots += f"[{player.symbol if slot else '.'}]"

        print(
            f"  {player.symbol} {player.name:10s} ({player.type_label:35s}) | "
            f"Koti: {home_count} | Rata: {track_count} | "
            f"Maali: {goal_slots}"
        )
    print("=" * 62)
    print("  *=lähtöruutu  S=Sininen P=Punainen K=Keltainen V=Vihreä")
