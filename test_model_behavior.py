#!/usr/bin/env python3
"""Test trained model behavior."""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.rl.env.tft_env import TFTEnv
from src.rl.env.action_space import ActionSpace, ActionType
from src.rl.models.custom_masked_ppo import CustomMaskedPPO
from src.rl.models.dueling_dqn import DuelingDQNModel
from src.rl.models.base import ModelConfig


def test_model(model_path: str, model_type: str = "ppo", n_episodes: int = 5):
    """Test model and show action distribution."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print(f"{'='*60}")

    # Create env
    env = TFTEnv(num_players=8, max_rounds=50)
    tft_action_space = ActionSpace()

    # Create and load model
    config = ModelConfig(
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hidden_dims=[256, 128],
        features_dim=256,
    )

    if model_type == "dqn":
        model = DuelingDQNModel(env, config)
    else:
        model = CustomMaskedPPO(env, config)

    model.load(model_path.replace(".pt", ""))
    print(f"Model loaded. Total timesteps trained: {model.total_timesteps:,}")

    # Run episodes
    all_actions = []
    placements = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_actions = []
        step = 0

        print(f"\n--- Episode {ep+1} ---")

        while not done:
            action_mask = info.get("valid_action_mask")
            action, _ = model.predict(obs, action_mask, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Decode action
            action_type, details = tft_action_space.decode_action(action)
            episode_actions.append(action_type.name)
            all_actions.append(action_type.name)
            step += 1

            # Show first 20 actions per episode
            if step <= 20:
                player = env.game.players[env.agent_player_idx]
                board_count = len(player.units.board) if player.units else 0
                bench_count = len([u for u in player.units.bench if u]) if player.units else 0
                hp = getattr(player, 'hp', getattr(player, 'health', 100))
                level = getattr(player, 'level', 1)
                xp = getattr(player, 'xp', 0)
                print(f"  Step {step:3d}: {action_type.name:15s} | Gold: {player.gold:3d} | Lv: {level} | XP: {xp:2d} | Board: {board_count} | Bench: {bench_count} | HP: {hp}")

        placement = info.get("placement", 8)
        placements.append(placement)

        # Episode action summary
        action_counts = Counter(episode_actions)
        print(f"\n  Episode {ep+1} Summary:")
        print(f"    Placement: {placement}")
        print(f"    Total steps: {step}")
        print(f"    Action distribution:")
        for action_name, count in action_counts.most_common():
            pct = count / step * 100
            print(f"      {action_name:15s}: {count:4d} ({pct:5.1f}%)")

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Average placement: {sum(placements)/len(placements):.2f}")
    print(f"Placements: {placements}")

    total = len(all_actions)
    action_counts = Counter(all_actions)
    print(f"\nAction distribution (all episodes):")
    for action_name, count in action_counts.most_common():
        pct = count / total * 100
        print(f"  {action_name:15s}: {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    # Test Phase 2 DuelingDQN model
    model_path = "models/league_phase2_20251227_173717/duelingdqnmodel_league_final.pt"
    test_model(model_path, model_type="dqn", n_episodes=2)
