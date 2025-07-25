import gym
import numpy as np
import torch
from Environment.env import PilotEnv
from Agent.dqn_agent import DQNAgent
import matplotlib.pyplot as plt

def train_dqn(num_episodes=500, max_steps=200, device='cpu'):
    env = PilotEnv(max_steps=max_steps, seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device=device)

    rewards_history = []
    avg_rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            if done:
                break

        rewards_history.append(total_reward)
        avg_reward = np.mean(rewards_history[-50:])
        avg_rewards_history.append(avg_reward)

        print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward:.2f} - Avg Reward: {avg_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'dqn_pilot_control_model.pth')

    # Plot learning curve
    plt.figure(figsize=(10,6))
    plt.plot(rewards_history, label='Episode Reward')
    plt.plot(avg_rewards_history, label='Average Reward (50 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Reward Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dqn(num_episodes=500, max_steps=200, device=device)
