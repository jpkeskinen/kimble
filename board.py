"""
Kimble-pelilauta.

Rakenne:
- 28 ruutua pääradalla (indeksit 0-27)
- Kullakin pelaajalla 4 maalialueen ruutua (erillinen lista)
- Kotipesässä odottavat nappulat (ei laudalla)

Pelaajien lähtöruudut pääradalla:
  Pelaaja 0 (sininen):  ruutu 0
  Pelaaja 1 (punainen): ruutu 7
  Pelaaja 2 (keltainen):ruutu 14
  Pelaaja 3 (vihreä):   ruutu 21

Nappulan tila:
  pos = -1         → kotipesässä
  pos = 0..27      → pääradalla (absoluuttinen indeksi)
  pos = 100+i      → maalialueella, ruutu i (0-3, 0=sisäänkäynti, 3=perä)
"""

TRACK_SIZE = 28
HOME_SIZE = 4
NUM_PLAYERS = 4

PLAYER_STARTS = [0, 7, 14, 21]   # lähtöruutu pääradalla
PLAYER_COLORS = ["Sininen", "Punainen", "Keltainen", "Vihreä"]
PLAYER_SYMBOLS = ["S", "P", "K", "V"]

# Ennen maalialueen sisäänkäyntiä oleva ruutu (pelaajan lähtö - 1, mod 28)
def home_entry(player: int) -> int:
    """Ruutu pääradalla juuri ennen maalialuetta."""
    return (PLAYER_STARTS[player] - 1) % TRACK_SIZE


class Piece:
    def __init__(self, player: int, idx: int):
        self.player = player
        self.idx = idx       # nappulan järjestysnumero (0-3)
        self.pos = -1        # -1 = kotipesässä

    def is_home(self) -> bool:
        return self.pos == -1

    def is_on_track(self) -> bool:
        return 0 <= self.pos < TRACK_SIZE

    def is_in_goal(self) -> bool:
        return self.pos >= 100

    def goal_index(self) -> int:
        """Maalialueen sisäinen indeksi (0-3), tai -1 jos ei maalissa."""
        if self.is_in_goal():
            return self.pos - 100
        return -1

    def __repr__(self):
        if self.is_home():
            state = "koti"
        elif self.is_in_goal():
            state = f"maali[{self.goal_index()}]"
        else:
            state = f"rata[{self.pos}]"
        return f"Piece({PLAYER_COLORS[self.player]}/{self.idx} @{state})"
