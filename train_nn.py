#!/usr/bin/env python3
"""Harjoittele Kimble-neuroverkko itsepelillä (REINFORCE-politiikkagradientti)."""

import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

from nn_player import KimbleNet, NNPlayer, MODEL_PATH
from board import NUM_PLAYERS
from player import Player
from game import Game


def collect_episode(nn: NNPlayer) -> tuple[list[list], int]:
    """
    Pelaa yksi peli 4 NN-pelaajalla (jaettu malli).
    Palauttaa (trajectories, winner_id).
    trajectories[i] on lista log_prob-tensoreista pelaajalle i.
    """
    players = [Player(i, is_human=False, strategy='nn') for i in range(NUM_PLAYERS)]
    for p in players:
        p._all_players = players
        p._nn_override = nn
        p._training_nn = True

    winner = Game(players, verbose=False).run()
    return [p._trajectory for p in players], winner


def evaluate(model: KimbleNet, device: str, n: int = 200) -> float:
    """
    Pelaa malli 3 longest_eat-vastustajaa vastaan.
    Palauttaa pelaajan 0 voittoprosenttiluvun.
    """
    nn = NNPlayer(model=model, device=device)
    nn.model.eval()
    wins = 0
    for _ in range(n):
        players = [Player(i, is_human=False, strategy='longest_eat') for i in range(NUM_PLAYERS)]
        players[0] = Player(0, is_human=False, strategy='nn')
        for p in players:
            p._all_players = players
        players[0]._nn_override = nn
        winner = Game(players, verbose=False).run()
        if winner == 0:
            wins += 1
    return wins / n


def train(num_episodes: int, batch_size: int, lr: float, device: str):
    model = KimbleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Juokseva baseline-arvio (lähestyy 0.25:tta)
    baseline = 0.25

    total_batches = num_episodes // batch_size
    log_every = max(1, 1000 // batch_size)
    save_every = max(1, 5000 // batch_size)

    print(f"Harjoitellaan {num_episodes} episodia, batch={batch_size}, lr={lr}, laite={device}")

    for batch_idx in tqdm(range(total_batches), desc="Harjoitellaan"):
        model.train()
        nn = NNPlayer(model=model, device=device)

        batch_losses: list["torch.Tensor"] = []

        for _ in range(batch_size):
            trajectories, winner = collect_episode(nn)

            for player_id, log_probs in enumerate(trajectories):
                if not log_probs:
                    continue
                reward = 1.0 if player_id == winner else 0.0
                advantage = reward - baseline
                for lp in log_probs:
                    batch_losses.append(-lp * advantage)

            # Päivitä baseline eksponentiaalisella liukuvalla keskiarvolla
            for pid in range(NUM_PLAYERS):
                r = 1.0 if pid == winner else 0.0
                baseline = 0.99 * baseline + 0.01 * r

        if batch_losses:
            optimizer.zero_grad()
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_val = loss.item()
        else:
            loss_val = 0.0

        episode_num = (batch_idx + 1) * batch_size

        if (batch_idx + 1) % log_every == 0:
            model.eval()
            win_rate = evaluate(model, device, n=200)
            tqdm.write(
                f"  episodit={episode_num:6d}  loss={loss_val:.4f}  "
                f"voitto vs longest_eat={win_rate:.1%}  baseline={baseline:.3f}"
            )

        if (batch_idx + 1) % save_every == 0:
            torch.save(model.state_dict(), MODEL_PATH)
            tqdm.write(f"  Tallennettu: {MODEL_PATH}")

    # Tallenna lopullinen malli
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nHarjoittelu valmis. Malli tallennettu: {MODEL_PATH}")

    model.eval()
    final_rate = evaluate(model, device, n=500)
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
    )


if __name__ == "__main__":
    main()
