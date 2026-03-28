#!/usr/bin/env python3
"""Harjoittele Kimble-neuroverkko (REINFORCE tai Actor-Critic + self-play)."""

import argparse
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import random

from nn_player import KimbleNet, KimbleDeepNet, KimbleActorCritic, NNPlayer, MODEL_PATH, DEEP_MODEL_PATH, AC_MODEL_PATH
from board import NUM_PLAYERS
from player import Player, STRATEGY_KEYS
from game import Game

# Strategiat joita käytetään satunnaisina vastustajina koulutuksessa (ei NN-strategioita)
_NN_STRATEGIES = {'nn', 'nn_deep'}
_OPPONENT_STRATEGIES = [s for s in STRATEGY_KEYS if s not in _NN_STRATEGIES]


def collect_selfplay_episode(
    nn: NNPlayer,
    strategy: str,
    opponent_pool: list,
    model_class,
    device: str,
) -> tuple[list[list], int]:
    """Self-play: 70% kaikki 4 NN-pelaajaa, 30% pelaaja 0 vs historiallisia malleja poolista."""
    use_pool = len(opponent_pool) > 0 and random.random() < 0.3
    players = []

    for i in range(NUM_PLAYERS):
        p = Player(i, is_human=False, strategy=strategy)
        if not use_pool or i == 0:
            p._nn_override = nn
            p._training_nn = True
        else:
            hist_model = model_class()
            hist_model.load_state_dict(random.choice(opponent_pool))
            hist_model.to(device)
            hist_model.eval()
            p._nn_override = NNPlayer(model=hist_model, device=device)
            p._training_nn = False
        players.append(p)

    for p in players:
        p._all_players = players

    winner = Game(players, verbose=False).run()
    return [p._trajectory for p in players], winner


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
    """Pelaa malli 3 longest_eat-vastustajaa vastaan. Palauttaa voittoprosentin."""
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


GAMMA = 0.99        # diskontauskerroin
ENTROPY_COEF = 0.01 # entropiabonus (kannustaa tutkimiseen)
CRITIC_COEF = 0.5   # critic-häviön paino


def train(num_episodes: int, batch_size: int, lr: float, device: str, deep: bool = False, ac: bool = False):
    if ac:
        net_class = KimbleActorCritic
        model_path = AC_MODEL_PATH
        strategy = 'nn_ac'
        model_label = "Actor-Critic (KimbleActorCritic) + self-play"
    elif deep:
        net_class = KimbleDeepNet
        model_path = DEEP_MODEL_PATH
        strategy = 'nn_deep'
        model_label = "syvä (KimbleDeepNet)"
    else:
        net_class = KimbleNet
        model_path = MODEL_PATH
        strategy = 'nn'
        model_label = "matala (KimbleNet)"

    model = net_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    opponent_pool: list = []  # historialliset mallit self-play-diversiteettiin

    total_batches = num_episodes // batch_size
    log_every = max(1, 1000 // batch_size)
    save_every = max(1, 5000 // batch_size)

    print(f"Harjoitellaan {model_label}, {num_episodes} episodia, batch={batch_size}, lr={lr}, laite={device}")
    print(f"  Reward shaping: eteneminen=0.005/yks, syöminen=+0.3, syödyksi=-0.2, maali=+0.5, gamma={GAMMA}")
    if ac:
        print(f"  AC-parametrit: critic_coef={CRITIC_COEF}, entropy_coef={ENTROPY_COEF}, opponent_pool=10")

    for batch_idx in tqdm(range(total_batches), desc="Harjoitellaan"):
        model.train()
        nn = NNPlayer(model=model, device=device)

        # Keräysrakenteet
        actor_log_probs: list["torch.Tensor"] = []
        actor_advantages: list["torch.Tensor"] = []  # G_t (REINFORCE) tai G_t - V(s) (AC)
        critic_values: list["torch.Tensor"] = []
        critic_targets: list["torch.Tensor"] = []
        entropy_terms: list["torch.Tensor"] = []

        for _ in range(batch_size):
            if ac:
                trajectories, winner = collect_selfplay_episode(
                    nn, strategy, opponent_pool, net_class, device
                )
            else:
                trajectories, winner = collect_episode(nn, strategy=strategy)

            for player_id, trajectory in enumerate(trajectories):
                if not trajectory:
                    continue
                terminal = 1.0 if player_id == winner else 0.0

                # Laske diskontattut palautukset taaksepäin: G_t = r_t + γ * G_{t+1}
                G = terminal
                ep_returns: list[float] = []
                for lp, sr, v, ent in reversed(trajectory):
                    G = sr + GAMMA * G
                    ep_returns.append(G)
                ep_returns.reverse()

                for (lp, _, v, ent), G_t in zip(trajectory, ep_returns):
                    G_tensor = torch.tensor(G_t, dtype=torch.float32, device=device)
                    actor_log_probs.append(lp)
                    if v is not None:
                        # Actor-Critic: per-tila advantage
                        actor_advantages.append(G_tensor - v.detach())
                        critic_values.append(v)
                        critic_targets.append(G_tensor)
                    else:
                        # REINFORCE: normalisoidaan myöhemmin
                        actor_advantages.append(G_tensor)
                    if ent is not None:
                        entropy_terms.append(ent)

        loss_val = 0.0
        if actor_log_probs:
            if ac:
                # Actor-Critic: per-tila advantage, ei normalisointia
                adv_stack = torch.stack(actor_advantages)
            else:
                # REINFORCE: normalisoi batch-tasolla varianssin pienentämiseksi
                adv_stack = torch.stack(actor_advantages)
                adv_stack = (adv_stack - adv_stack.mean()) / (adv_stack.std() + 1e-8)

            actor_loss = torch.stack(
                [-lp * adv for lp, adv in zip(actor_log_probs, adv_stack)]
            ).mean()
            loss = actor_loss

            if critic_values:
                critic_loss = F.mse_loss(
                    torch.stack(critic_values),
                    torch.stack(critic_targets),
                )
                loss = loss + CRITIC_COEF * critic_loss

            if entropy_terms:
                loss = loss - ENTROPY_COEF * torch.stack(entropy_terms).mean()

            optimizer.zero_grad()
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
            if ac:
                opponent_pool.append(copy.deepcopy(model.state_dict()))
                opponent_pool = opponent_pool[-10:]  # pidä viimeiset 10
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
    parser.add_argument("--ac", action="store_true",
                        help="Harjoittele Actor-Critic (KimbleActorCritic) + self-play")
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
        ac=args.ac,
    )


if __name__ == "__main__":
    main()
