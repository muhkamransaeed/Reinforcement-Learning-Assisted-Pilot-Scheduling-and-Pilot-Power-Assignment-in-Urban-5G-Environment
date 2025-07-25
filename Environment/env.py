import gym
from gym import spaces
import numpy as np
from Channel_Modeling.urban_channel import UrbanChannelModel

class PilotEnv(gym.Env):
    """
    Custom environment for pilot scheduling and pilot power control
    in a dynamic urban wireless channel.
    """

    def __init__(self,
                 max_steps=200,
                 pilot_powers=[0.0, 0.1, 0.5, 1.0],  # in Watts
                 carrier_freq_ghz=2.5,
                 user_speed_mps=3,
                 seed=None):
        super(PilotEnv, self).__init__()

        self.max_steps = max_steps
        self.pilot_powers = pilot_powers
        self.n_powers = len(pilot_powers)
        self.carrier_freq_ghz = carrier_freq_ghz
        self.user_speed_mps = user_speed_mps

        # Action space:
        # 0: no pilot sent (power=0)
        # 1..n_powers-1: send pilot with corresponding power
        self.action_space = spaces.Discrete(self.n_powers)

        # State space: 7 continuous features normalized between 0 and 1
        # Features:
        # [SINR, Doppler (normalized), Time since last pilot, Previous NMSE,
        #  Previous pilot power, Mobility speed (normalized), Channel age]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        # Initialize channel model
        self.channel_model = UrbanChannelModel(
            carrier_freq_ghz=self.carrier_freq_ghz,
            user_speed_mps=self.user_speed_mps,
            seed=seed
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count = 0
        self.time_since_last_pilot = 1  # start counting from 1 to avoid div by zero
        self.prev_nmse = 1.0  # start with high NMSE (bad estimation)
        self.prev_pilot_power = 0.0
        self.channel_age = 0

        # Reset urban channel model state (position, fading, etc.)
        self.channel_model.reset()

        # Initial channel stats
        self.current_sinr = self.channel_model.get_sinr()
        self.current_doppler = self.channel_model.get_normalized_doppler()

        return self._get_state()

    def step(self, action):
        """
        Executes one time step in the environment.

        Parameters:
            action (int): discrete action index

        Returns:
            state, reward, done, info
        """
        self.step_count += 1
        done = self.step_count >= self.max_steps

        # Decode action
        pilot_power = self.pilot_powers[action]

        # Update time since last pilot
        if pilot_power > 0.0:
            self.time_since_last_pilot = 1
            self.channel_age = 0
        else:
            self.time_since_last_pilot += 1
            self.channel_age += 1

        # Simulate channel fading and update SINR, Doppler etc.
        self.channel_model.step(user_speed=self.user_speed_mps,
                                pilot_power=pilot_power,
                                time_since_last_pilot=self.time_since_last_pilot)

        # Get updated channel stats
        self.current_sinr = self.channel_model.get_sinr()
        self.current_doppler = self.channel_model.get_normalized_doppler()
        self.prev_nmse = self.channel_model.get_nmse()
        self.prev_pilot_power = pilot_power

        # Calculate reward
        reward = self._compute_reward()

        # Construct next state
        next_state = self._get_state()

        info = {
            'sinr': self.current_sinr,
            'doppler': self.current_doppler,
            'nmse': self.prev_nmse,
            'pilot_power': pilot_power,
            'time_since_last_pilot': self.time_since_last_pilot,
        }

        return next_state, reward, done, info

    def _get_state(self):
        """
        Normalize and return current state as vector.
        """

        # Normalize values to [0, 1] (simple min-max scaling)
        sinr_norm = np.clip(self.current_sinr / 30.0, 0.0, 1.0)  # assuming max SINR ~30 dB
        doppler_norm = self.current_doppler  # already normalized by channel model
        time_pilot_norm = np.clip(self.time_since_last_pilot / 50.0, 0.0, 1.0)
        nmse_norm = np.clip(self.prev_nmse, 0.0, 1.0)
        pilot_power_norm = np.clip(self.prev_pilot_power / max(self.pilot_powers), 0.0, 1.0)
        mobility_norm = np.clip(self.user_speed_mps / 30.0, 0.0, 1.0)  # max speed 30 m/s
        channel_age_norm = np.clip(self.channel_age / 50.0, 0.0, 1.0)

        return np.array([
            sinr_norm,
            doppler_norm,
            time_pilot_norm,
            nmse_norm,
            pilot_power_norm,
            mobility_norm,
            channel_age_norm,
        ], dtype=np.float32)

    def _compute_reward(self):
        """
        Reward balances throughput (log(1+SINR)),
        pilot overhead (penalty if pilot sent),
        pilot power cost, and NMSE penalty.
        """

        throughput = np.log2(1 + self.current_sinr)
        alpha = 0.5  # pilot overhead penalty weight
        beta = 0.5   # pilot power penalty weight
        gamma = 1.0  # NMSE penalty weight

        pilot_overhead = 1.0 if self.prev_pilot_power > 0 else 0.0
        pilot_power_cost = self.prev_pilot_power
        nmse_penalty = self.prev_nmse

        reward = throughput - alpha * pilot_overhead - beta * pilot_power_cost - gamma * nmse_penalty
        return reward
