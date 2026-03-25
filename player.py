"""Pelaaja-luokat: ihminen ja tietokone."""
import random
from board import Piece, PLAYER_COLORS, PLAYER_SYMBOLS, PLAYER_STARTS, TRACK_SIZE

STRATEGY_NAMES = {
    'random':            'Satunnainen',
    'longest':           'Pisin ensin (kuutosella uusi nappula)',
    'longest_eat':       'Pisin ensin + syö',
    'longest_eat_dodge': 'Pisin ensin + syö + väistä',
    'shortest':          'Lyhin ensin',
    'shortest_eat':      'Lyhin ensin + syö',
    'nn':                'Neuroverkko',
}
STRATEGY_KEYS = list(STRATEGY_NAMES.keys())

_EAT_STRATEGIES = {'longest_eat', 'longest_eat_dodge', 'shortest_eat'}
_LONGEST_STRATEGIES = {'longest', 'longest_eat', 'longest_eat_dodge'}
_DEFENSE_STRATEGIES = {'longest_eat_dodge'}


class Player:
    def __init__(self, player_id: int, is_human: bool, strategy: str = 'random'):
        self.player_id = player_id
        self.is_human = is_human
        self.strategy = strategy
        self.name = PLAYER_COLORS[player_id]
        self.symbol = PLAYER_SYMBOLS[player_id]
        self.start = PLAYER_STARTS[player_id]
        self.pieces = [Piece(player_id, i) for i in range(4)]
        self._just_launched: Piece | None = None
        # Neuroverkkostrategiaa varten
        self._all_players: list | None = None    # asetetaan Game.__init__:ssa
        self._nn_override: object | None = None  # NNPlayer-instanssi harjoittelua varten
        self._training_nn: bool = False          # True → tallenna log_probit
        self._trajectory: list = []              # log_prob-lista episodille

    @property
    def type_label(self) -> str:
        if self.is_human:
            return "ihminen"
        return STRATEGY_NAMES.get(self.strategy, self.strategy)

    @property
    def uses_eating(self) -> bool:
        return self.strategy in _EAT_STRATEGIES

    @property
    def uses_defense(self) -> bool:
        return self.strategy in _DEFENSE_STRATEGIES

    def pieces_at_home(self):
        return [p for p in self.pieces if p.is_home()]

    def pieces_on_track(self):
        return [p for p in self.pieces if p.is_on_track()]

    def pieces_in_goal(self):
        return [p for p in self.pieces if p.is_in_goal()]

    def has_won(self) -> bool:
        return len(self.pieces_in_goal()) == 4

    def on_piece_moved(self, piece: Piece, was_home: bool) -> None:
        """Päivitä sisäinen tila siirron jälkeen (strategian bonus-heittoa varten)."""
        if was_home and self.strategy in _LONGEST_STRATEGIES:
            self._just_launched = piece
        else:
            self._just_launched = None

    def _piece_distance(self, piece: Piece) -> int:
        """Kuinka pitkälle nappula on liikkunut lähtöruudusta."""
        if piece.is_home():
            return 0
        if piece.is_on_track():
            return (piece.pos - self.start + TRACK_SIZE) % TRACK_SIZE
        return TRACK_SIZE + piece.goal_index() + 1

    def choose_piece(
        self,
        movable: list[Piece],
        die: int,
        can_eat: set | None = None,
        threatened: set | None = None,
    ) -> Piece | None:
        """Valitse siirrettävä nappula."""
        if not movable:
            return None
        if self.is_human:
            return self._human_choose(movable, die)
        return self._ai_choose(movable, die, can_eat or set(), threatened or set())

    def _human_choose(self, movable: list[Piece], die: int) -> Piece:
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

    def _ai_choose(self, movable: list[Piece], die: int, can_eat: set, threatened: set) -> Piece:
        # Neuroverkkostrategia
        if self.strategy == 'nn':
            from nn_player import get_nn_player
            nn = self._nn_override if self._nn_override is not None else get_nn_player()
            all_players = self._all_players or []
            result = nn.choose_piece(
                self, movable, die, all_players, can_eat, threatened,
                training=self._training_nn,
            )
            if self._training_nn:
                piece, log_prob = result
                if log_prob is not None:
                    self._trajectory.append(log_prob)
                return piece if piece is not None else random.choice(movable)
            return result if result is not None else random.choice(movable)

        # Syöminen on ylin prioriteetti strategioille longest_eat ja shortest_eat
        if self.strategy in _EAT_STRATEGIES and can_eat:
            best = (max if self.strategy == 'longest_eat' else min)(
                (p for p in movable if p in can_eat), default=None, key=self._piece_distance
            )
            if best is not None:
                return best

        # Väistä uhattua nappulaa
        if self.strategy in _DEFENSE_STRATEGIES and threatened:
            candidates = [p for p in movable if p in threatened]
            if candidates:
                return max(candidates, key=self._piece_distance)

        if self.strategy == 'random':
            return random.choice(movable)

        if self.strategy in _LONGEST_STRATEGIES:
            # Bonus-heitolla suositaan juuri kotipesästä otettua nappulaa
            just_launched, self._just_launched = self._just_launched, None
            if just_launched is not None and just_launched in movable:
                return just_launched
            # Kuutosella otetaan uusi nappula kotipesästä
            if die == 6:
                home = next((p for p in movable if p.is_home()), None)
                if home is not None:
                    return home
            return max(movable, key=self._piece_distance)

        # shortest ja shortest_eat (syöminen käsitelty jo yllä)
        return min(movable, key=self._piece_distance)
