import matplotlib.pyplot as plt
import numpy as np

def plot_training_curve(rewards, avg_rewards, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (window=50)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Reward Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_evaluation_summary(summary, save_path=None):
    policies = list(summary.keys())
    metrics = ['avg_reward', 'avg_throughput', 'avg_nmse', 'avg_pilot_power', 'pilot_fraction']

    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

    for idx, metric in enumerate(metrics):
        values = [summary[p][metric] for p in policies]
        axs[idx].bar(policies, values, color='skyblue')
        axs[idx].set_title(metric.replace('_', ' ').title())
        axs[idx].set_ylabel(metric.replace('_', ' ').title())
        axs[idx].grid(axis='y')

        # Show value on top of bars
        for i, v in enumerate(values):
            axs[idx].text(i, v + 0.01 * max(values), f"{v:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_time_series(data_dict, metric_name, xlabel='Time Step', ylabel=None, save_path=None):
    plt.figure(figsize=(10,6))
    for label, data in data_dict.items():
        plt.plot(data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or metric_name)
    plt.title(f"{metric_name} Over Time")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


# Example usage:
if __name__ == "__main__":
    # For training curve
    # You can load rewards from a saved file or pass directly
    # plot_training_curve(rewards, avg_rewards)

    # For evaluation summary
    example_summary = {
        'dqn': {'avg_reward': 120, 'avg_throughput': 2.5, 'avg_nmse': 0.1, 'avg_pilot_power': 0.3, 'pilot_fraction': 0.25},
        'fixed_interval': {'avg_reward': 90, 'avg_throughput': 2.0, 'avg_nmse': 0.15, 'avg_pilot_power': 0.5, 'pilot_fraction': 0.4},
        'always_on': {'avg_reward': 80, 'avg_throughput': 1.8, 'avg_nmse': 0.05, 'avg_pilot_power': 1.0, 'pilot_fraction': 1.0},
        'random': {'avg_reward': 60, 'avg_throughput': 1.5, 'avg_nmse': 0.3, 'avg_pilot_power': 0.4, 'pilot_fraction': 0.3}
    }
    plot_evaluation_summary(example_summary)
