"""Neuroverkko-tekoälystrategia Kimble-peliin (KimbleNet)."""

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import os
import random

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nn_model.pt")
_NORM = 32.0  # _piece_distance() maksimiarvo (TRACK_SIZE + HOME_SIZE)


def build_state_vector(
    player,
    movable: list,
    die: int,
    all_players: list,
    can_eat: set,
    threatened: set,
) -> "torch.Tensor":
    """Koodaa pelitila 34-elementtiseksi tensoriksi (nykyisen pelaajan näkökulmasta)."""
    features: list[float] = []

    # Omat 4 nappulaa: normalisoitu edistyminen (4 piirrettä)
    for piece in player.pieces:
        features.append(player._piece_distance(piece) / _NORM)

    # 3 vastustajaa myötäpäivään pelaajasta katsottuna (12 piirrettä)
    n = len(all_players)
    for offset in range(1, n):
        opp = all_players[(player.player_id + offset) % n]
        for piece in opp.pieces:
            features.append(opp._piece_distance(piece) / _NORM)

    # Nopan silmäluku one-hot (6 piirrettä)
    for d in range(1, 7):
        features.append(1.0 if die == d else 0.0)

    # Liikutettavat nappulat -maski (4 piirrettä)
    movable_ids = {id(p) for p in movable}
    for piece in player.pieces:
        features.append(1.0 if id(piece) in movable_ids else 0.0)

    # Syömismahdollisuus-maski (4 piirrettä)
    eat_ids = {id(p) for p in can_eat}
    for piece in player.pieces:
        features.append(1.0 if id(piece) in eat_ids else 0.0)

    # Uhattuna olevat nappulat -maski (4 piirrettä)
    threat_ids = {id(p) for p in threatened}
    for piece in player.pieces:
        features.append(1.0 if id(piece) in threat_ids else 0.0)

    # Yhteensä: 4 + 12 + 6 + 4 + 4 + 4 = 34 piirrettä
    return torch.tensor(features, dtype=torch.float32)


class KimbleNet(nn.Module):
    """Matala MLP-verkko: 34 → 128 → 64 → 4 (logitit jokaiselle nappulalle)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class NNPlayer:
    """Nappulanvalitsija, joka käyttää KimbleNet-verkkoa."""

    def __init__(self, model: "KimbleNet | None" = None, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch ei ole asennettuna. Asenna: pip install torch"
            )
        self.device = device
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = self._load_model()
        if self.model is not None:
            self.model.eval()

    def _load_model(self) -> "KimbleNet | None":
        if not os.path.exists(MODEL_PATH):
            print(
                f"[NNPlayer] Mallia ei löydy polusta: {MODEL_PATH}\n"
                "  Harjoittele malli ensin: python train_nn.py"
            )
            return None
        m = KimbleNet()
        m.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        return m.to(self.device)

    def choose_piece(
        self,
        player,
        movable: list,
        die: int,
        all_players: list,
        can_eat: set,
        threatened: set,
        training: bool = False,
    ):
        """
        Valitse nappula.
        - inference: palauttaa Piece
        - training: palauttaa (Piece, log_prob_tensori)
        """
        if not movable:
            return (None, None) if training else None

        if self.model is None:
            piece = random.choice(movable)
            if training:
                return piece, torch.tensor(0.0, dtype=torch.float32)
            return piece

        state = build_state_vector(
            player, movable, die, all_players, can_eat, threatened
        )
        state = state.to(self.device)

        with torch.set_grad_enabled(training):
            logits = self.model(state)  # muoto (4,)

            # Maskaa ei-liikutettavat nappulat hyvin pienellä arvolla
            movable_ids = {id(p) for p in movable}
            mask = torch.tensor(
                [1.0 if id(p) in movable_ids else 0.0 for p in player.pieces],
                dtype=torch.float32,
                device=self.device,
            )
            masked_logits = logits + (mask - 1.0) * 1e9

        if training:
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            piece = player.pieces[int(action.item())]
            if piece not in movable:
                piece = random.choice(movable)
            return piece, log_prob
        else:
            piece_idx = int(torch.argmax(masked_logits).item())
            piece = player.pieces[piece_idx]
            if piece not in movable:
                piece = random.choice(movable)
            return piece


_cached_nn_player: "NNPlayer | None" = None


def get_nn_player() -> "NNPlayer":
    """Palauta jaettu NNPlayer-instanssi (ladataan tarvittaessa)."""
    global _cached_nn_player
    if _cached_nn_player is None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch ei ole asennettuna.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cached_nn_player = NNPlayer(device=device)
    return _cached_nn_player
