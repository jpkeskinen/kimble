#!/usr/bin/env python3
"""Harjoittele Kimble-neuroverkko itsepelillä (REINFORCE-politiikkagradientti)."""

import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

import random

from nn_player import KimbleNet, KimbleDeepNet, NNPlayer, MODEL_PATH, DEEP_MODEL_PATH
from board import NUM_PLAYERS
from player import Player, STRATEGY_KEYS
from game import Game

# Strategiat joita käytetään satunnaisina vastustajina koulutuksessa (ei NN-strategioita)
_NN_STRATEGIES = {'nn', 'nn_deep'}
_OPPONENT_STRATEGIES = [s for s in STRATEGY_KEYS if s not in _NN_STRATEGIES]


def collect_episode(nn: NNPlayer, strategy: str = 'nn') -> tuple[list[list], int]:
    """
    Pelaa yksi peli: pelaaja 0 on NN (strategy='nn' tai 'nn_deep'), muut saavat satunnaisen strategian.
    Palauttaa (trajectories, winner_id).
    trajectories[0] on lista log_prob-tensoreista NN-pelaajalle (muut ovat tyhjiä).
    """
    players = []
    for i in range(NUM_PLAYERS):
        if i == 0:
            p = Player(i, is_human=False, strategy=strategy)
        else:
            s = random.choice(_OPPONENT_STRATEGIES)
            p = Player(i, is_human=False, strategy=s)
        players.append(p)

    for p in players:
        p._all_players = players

    players[0]._nn_override = nn
    players[0]._training_nn = True

    winner = Game(players, verbose=False).run()
    return [p._trajectory for p in players], winner


def evaluate(model, device: str, strategy: str = 'nn', n: int = 200) -> float:
    """
    Pelaa malli 3 longest_eat-vastustajaa vastaan.
    Palauttaa pelaajan 0 voittoprosenttiluvun.
    """
    nn = NNPlayer(model=model, device=device)
    nn.model.eval()
    wins = 0
    for _ in range(n):
        players = [Player(i, is_human=False, strategy='longest_eat') for i in range(NUM_PLAYERS)]
        players[0] = Player(0, is_human=False, strategy=strategy)
        for p in players:
            p._all_players = players
        players[0]._nn_override = nn
        winner = Game(players, verbose=False).run()
        if winner == 0:
            wins += 1
    return wins / n


GAMMA = 0.99  # diskontauskerroin


def train(num_episodes: int, batch_size: int, lr: float, device: str, deep: bool = False):
    net_class = KimbleDeepNet if deep else KimbleNet
    model_path = DEEP_MODEL_PATH if deep else MODEL_PATH
    strategy = 'nn_deep' if deep else 'nn'
    model_label = "syvä (KimbleDeepNet)" if deep else "matala (KimbleNet)"

    model = net_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_batches = num_episodes // batch_size
    log_every = max(1, 1000 // batch_size)
    save_every = max(1, 5000 // batch_size)

    print(f"Harjoitellaan {model_label}, {num_episodes} episodia, batch={batch_size}, lr={lr}, laite={device}")
    print(f"  Reward shaping: eteneminen=0.005/yks, syöminen=+0.3, syödyksi=-0.2, maali=+0.5, gamma={GAMMA}")

    for batch_idx in tqdm(range(total_batches), desc="Harjoitellaan"):
        model.train()
        nn = NNPlayer(model=model, device=device)

        all_log_probs: list["torch.Tensor"] = []
        all_returns: list[float] = []

        for _ in range(batch_size):
            trajectories, winner = collect_episode(nn, strategy=strategy)

            for player_id, trajectory in enumerate(trajectories):
                if not trajectory:
                    continue
                terminal = 1.0 if player_id == winner else 0.0

                # Laske diskontattut palautukset taaksepäin: G_t = r_t + γ * G_{t+1}
                G = terminal
                ep_returns: list[float] = []
                for lp, sr in reversed(trajectory):
                    G = sr + GAMMA * G
                    ep_returns.append(G)
                ep_returns.reverse()

                for (lp, _), G in zip(trajectory, ep_returns):
                    all_log_probs.append(lp)
                    all_returns.append(G)

        loss_val = 0.0
        if all_log_probs:
            returns_t = torch.tensor(all_returns, dtype=torch.float32, device=device)
            # Normalisoi advantaget (vähentää varianssia)
            advantages = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
            batch_losses = [-lp * adv for lp, adv in zip(all_log_probs, advantages)]

            optimizer.zero_grad()
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_val = loss.item()

        episode_num = (batch_idx + 1) * batch_size

        if (batch_idx + 1) % log_every == 0:
            model.eval()
            win_rate = evaluate(model, device, strategy=strategy, n=200)
            tqdm.write(
                f"  episodit={episode_num:6d}  loss={loss_val:.4f}  "
                f"voitto vs longest_eat={win_rate:.1%}"
            )

        if (batch_idx + 1) % save_every == 0:
            torch.save(model.state_dict(), model_path)
            tqdm.write(f"  Tallennettu: {model_path}")

    # Tallenna lopullinen malli
    torch.save(model.state_dict(), model_path)
    print(f"\nHarjoittelu valmis. Malli tallennettu: {model_path}")

    model.eval()
    final_rate = evaluate(model, device, strategy=strategy, n=500)
    print(f"Lopullinen voittoprosentti vs longest_eat (n=500): {final_rate:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Harjoittele Kimble-neuroverkko")
    parser.add_argument("--episodes", type=int, default=50_000,
                        help="Harjoitteluepodien kokonaismäärä (oletus: 50000)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Episodeja per parametripäivitys (oletus: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Oppimisaste (oletus: 0.001)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Laite: auto, cpu tai cuda (oletus: auto)")
    parser.add_argument("--deep", action="store_true",
                        help="Harjoittele syvä verkko (KimbleDeepNet) matalan sijaan")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Käytetään laitetta: {device}")

    train(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        deep=args.deep,
    )


if __name__ == "__main__":
    main()
