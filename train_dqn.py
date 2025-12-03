import argparse
import os
import time
from typing import List

import numpy as np

from rl_env import TradingEnv
from dqn_agent import DQNAgent, DQNConfig


def train(symbols: List[str], start: str, end: str, window: int, episodes: int, steps_per_episode: int, 
          out_path: str, use_dueling: bool = True, use_per: bool = True):
    """
    Train Enhanced DQN agent with Dueling architecture and Prioritized Experience Replay.
    
    Args:
        symbols: List of stock symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        window: Observation window (days)
        episodes: Number of training episodes
        steps_per_episode: Steps per episode
        out_path: Path to save model
        use_dueling: Enable Dueling architecture
        use_per: Enable Prioritized Experience Replay
    """
    print("\n" + "=" * 70)
    print("DQN AGENT TRAINING - REAL DATA VERIFICATION")
    print("=" * 70)
    print(f"Training Period: {start} to {end}")
    print(f"Data Source: yfinance (REAL MARKET DATA)")
    print(f"Symbols: {len(symbols)}")
    print(f"Interval: Daily (1d)")
    print(f"Architecture: Dueling DQN with PER")
    print("=" * 70 + "\n")
    
    envs = [TradingEnv(sym, start=start, end=end, window=window) for sym in symbols]
    state_dim = envs[0].observation_space_shape[0]
    action_dim = envs[0].action_space_n
    
    print(f"[info] Loaded {len(symbols)} symbols with real yfinance data")
    
    # Display data verification for each environment
    print("\nData Verification:")
    total_bars = 0
    for env in envs:
        info = env.get_data_info()
        print(f"  [ok] {info['symbol']:6s}: {info['num_bars']:4d} bars from {info['date_range']} ({info['source']})")
        total_bars += info['num_bars']
    print(f"\n[ok] Total bars loaded: {total_bars:,}")
    
    print(f"\n[info] Training Enhanced DQN with Dueling={use_dueling}, PER={use_per}")
    print(f"[info] State dim: {state_dim}, Action dim: {action_dim}")
    print(f"[info] Episodes: {episodes}, Steps/ep: {steps_per_episode}\n")

    cfg = DQNConfig(
        gamma=0.99,
        lr=5e-4,
        batch_size=64,
        buffer_size=500_000,
        start_epsilon=1.0,
        end_epsilon=0.05,
        epsilon_decay_steps=200_000,
        target_update_interval=5000,
        hidden=256,
        device="cpu",
        use_dueling=use_dueling,
        use_per=use_per,
        per_alpha=0.6,
        per_beta=0.4
    )
    agent = DQNAgent(state_dim, action_dim, cfg)

    global_step = 0
    episode_rewards = []
    
    for ep in range(episodes):
        ep_reward = 0.0
        for env in envs:
            s = env.reset()
            for t in range(steps_per_episode):
                a = agent.act(s)
                s2, r, done, info = env.step(a)
                agent.push(s, a, r, s2, done)
                loss = agent.learn()
                s = s2
                ep_reward += r
                global_step += 1
                if done:
                    break
        
        episode_rewards.append(ep_reward)
        
        if (ep + 1) % 5 == 0 or ep == 0:
            avg_reward = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else ep_reward
            epsilon = agent._current_epsilon()
            buffer_size = len(agent.buffer)
            print(f"[ep {ep+1:3d}] Reward: {ep_reward:7.2f} | Avg(5): {avg_reward:7.2f} | Epsilon: {epsilon:.4f} | Buffer: {buffer_size}")

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    agent.save(out_path)
    print(f"\n[info] Saved DQN agent to {out_path}")
    print(f"[info] Final episode reward: {episode_rewards[-1]:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, default="AAPL,SPY,MSFT,GOOGL,NVDA,TSLA,META,NFLX,AMD,INTC,AVGO,ADBE,CSCO,CRM,DXCM,QQQ,IWM,GLD,USO,XLF,XLV,XLE,XLY,XLI,COIN,RIOT,MARA,SOXL,TQQQ,UPRO")
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="2025-12-02")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--out", type=str, default="models/dqn_agent.pt")
    p.add_argument("--use-dueling", type=bool, default=True, help="Enable Dueling architecture")
    p.add_argument("--use-per", type=bool, default=True, help="Enable Prioritized Experience Replay")
    
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    train(symbols, args.start, args.end, args.window, args.episodes, args.steps, args.out,
          use_dueling=args.use_dueling, use_per=args.use_per)

