import numpy as np
import torch
from Environment.env import PilotEnv
from Agent.dqn_agent import DQNAgent

def run_episode(env, agent=None, policy='dqn', max_steps=200):
    state = env.reset()
    total_reward = 0
    total_pilot_power = 0
    total_throughput = 0
    nmse_list = []
    pilot_transmissions = 0

    for step in range(max_steps):
        if policy == 'dqn':
            action = agent.select_action(state)
        elif policy == 'fixed_interval':
            # Send pilot every 10 steps at max power
            if step % 10 == 0:
                action = env.action_space.n - 1  # max power index
            else:
                action = 0  # no pilot
        elif policy == 'always_on':
            # Always send pilot with max power
            action = env.action_space.n - 1
        elif policy == 'random':
            action = env.action_space.sample()
        else:
            raise ValueError("Unknown policy")

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        total_pilot_power += info['pilot_power']
        total_throughput += np.log2(1 + info['sinr'])
        nmse_list.append(info['nmse'])
        if action != 0:
            pilot_transmissions += 1
        state = next_state
        if done:
            break

    avg_nmse = np.mean(nmse_list)
    avg_throughput = total_throughput / max_steps
    avg_pilot_power = total_pilot_power / max_steps
    pilot_fraction = pilot_transmissions / max_steps

    return {
        'total_reward': total_reward,
        'avg_nmse': avg_nmse,
        'avg_throughput': avg_throughput,
        'avg_pilot_power': avg_pilot_power,
        'pilot_fraction': pilot_fraction
    }

def evaluate(num_episodes=50, max_steps=200, device='cpu'):
    env = PilotEnv(max_steps=max_steps, seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device=device)
    agent.policy_net.load_state_dict(torch.load('dqn_pilot_control_model.pth', map_location=device))
    agent.policy_net.eval()

    policies = ['dqn', 'fixed_interval', 'always_on', 'random']
    results = {p: [] for p in policies}

    for ep in range(num_episodes):
        for p in policies:
            res = run_episode(env, agent if p == 'dqn' else None, policy=p, max_steps=max_steps)
            results[p].append(res)

    # Aggregate results
    summary = {}
    for p in policies:
        summary[p] = {
            'avg_reward': np.mean([r['total_reward'] for r in results[p]]),
            'avg_nmse': np.mean([r['avg_nmse'] for r in results[p]]),
            'avg_throughput': np.mean([r['avg_throughput'] for r in results[p]]),
            'avg_pilot_power': np.mean([r['avg_pilot_power'] for r in results[p]]),
            'pilot_fraction': np.mean([r['pilot_fraction'] for r in results[p]])
        }

    # Print summary
    print("Evaluation Summary (averaged over {} episodes):".format(num_episodes))
    for p in policies:
        print(f"\nPolicy: {p}")
        for k, v in summary[p].items():
            print(f"  {k}: {v:.4f}")

    return summary

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluate(num_episodes=50, max_steps=200, device=device)
