import numpy as np

class UrbanChannelModel:
    """
    Simulates urban wireless channel with:
    - Path loss (3GPP UMa simplified)
    - Doppler spread based on user speed
    - Small-scale fading (Rayleigh or Rician)
    - NMSE (channel estimation error) based on pilot power and age
    """

    def __init__(self,
                 carrier_freq_ghz=2.5,
                 user_speed_mps=3.0,
                 K_dB=6,             # Rician K-factor (set to 0 for Rayleigh)
                 seed=None):
        self.carrier_freq_ghz = carrier_freq_ghz
        self.user_speed_mps = user_speed_mps
        self.K_dB = K_dB
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.c = 3e8  # speed of light (m/s)

        # Initialize channel parameters
        self.distance_2d = 100.0  # meters, fixed UE-BS distance (can be randomized)
        self.h_bs = 25.0  # base station height (meters)
        self.h_ue = 1.5   # user equipment height (meters)

        self.reset()

    def reset(self):
        """
        Reset channel state, fading, and time
        """
        self.time = 0
        self.channel_age = 0

        # Initial fading sample
        self.h = self._generate_fading()

        # Initialize doppler_norm here to avoid AttributeError
        self.doppler_norm = self._doppler_spread(self.user_speed_mps, self.carrier_freq_ghz) / 200.0
        self.doppler_norm = max(0, min(self.doppler_norm, 1))
        
        # Initial SINR, NMSE placeholders
        self.sinr = 10.0  # initial SINR in linear scale (~10 dB)
        self.nmse = 1.0   # start with poor channel estimate

    def step(self, user_speed=None, pilot_power=0.0, time_since_last_pilot=1):
        """
        Simulate channel evolution for one time step.

        Args:
            user_speed (float): current user speed in m/s
            pilot_power (float): pilot transmit power in Watts
            time_since_last_pilot (int): slots since last pilot transmission
        """

        if user_speed is not None:
            self.user_speed_mps = user_speed

        # Update time and channel age
        self.time += 1
        self.channel_age = time_since_last_pilot

        # Update fading
        self.h = self._generate_fading()

        # Calculate path loss (dB)
        pl_dB = self._path_loss_uma(self.distance_2d,
                                    self.h_bs,
                                    self.h_ue,
                                    self.carrier_freq_ghz)

        # Calculate Doppler spread normalized [0,1]
        self.doppler_norm = self._doppler_spread(self.user_speed_mps,
                                                 self.carrier_freq_ghz) / 200.0
        self.doppler_norm = np.clip(self.doppler_norm, 0, 1)

        # Calculate instantaneous channel gain (linear scale)
        fading_gain = np.abs(self.h)**2
        path_loss_linear = 10**(-pl_dB / 10)

        # Received power proportional to pilot power * channel gain * path loss
        rx_power = pilot_power * fading_gain * path_loss_linear

        # Noise power (assumed fixed)
        noise_power = 1e-9  # Watts, adjust as needed

        # Calculate instantaneous SINR (linear scale)
        self.sinr = rx_power / noise_power

        # NMSE model:
        # Better estimation with more pilot power & fresher CSI
        # NMSE decreases exponentially with pilot power and pilot frequency
        if pilot_power > 0:
            self.nmse = 0.1 / (pilot_power + 1e-6)  # inverse relation to pilot power
            self.nmse *= np.exp(0.1 * self.channel_age)  # grows with channel age
            self.nmse = min(max(self.nmse, 1e-4), 1.0)
        else:
            # If no pilot, NMSE worsens exponentially over time
            self.nmse *= np.exp(0.1 * self.channel_age)
            self.nmse = min(self.nmse, 1.0)

    def get_sinr(self):
        # Return SINR in dB capped between 0-30 dB and normalized to linear scale
        sinr_db = 10 * np.log10(self.sinr + 1e-9)
        sinr_db = np.clip(sinr_db, 0, 30)
        return 10**(sinr_db / 10)

    def get_nmse(self):
        return self.nmse

    def get_normalized_doppler(self):
        # Normalized Doppler in [0,1] based on max expected Doppler ~200 Hz
        return self.doppler_norm

    def _path_loss_uma(self, d_2d, h_bs, h_ue, fc_ghz):
        """
        Simplified 3GPP Urban Macro (UMa) LOS path loss model (dB)
        """
        d_3d = np.sqrt(d_2d**2 + (h_bs - h_ue)**2)
        pl = 28 + 22 * np.log10(d_3d) + 20 * np.log10(fc_ghz)
        return pl

    def _doppler_spread(self, speed_mps, fc_ghz):
        """
        Doppler spread (Hz) = (v/c) * fc
        Max Doppler freq normalized to 200 Hz for convenience
        """
        doppler_hz = (speed_mps / self.c) * (fc_ghz * 1e9)
        return doppler_hz

    def _generate_fading(self):
        """
        Generate fading coefficient (complex):
        Rician if K_dB > 0 else Rayleigh
        """
        if self.K_dB <= 0:
            # Rayleigh fading
            real = self.np_random.normal(0, 1)
            imag = self.np_random.normal(0, 1)
            return (real + 1j * imag) / np.sqrt(2)
        else:
            # Rician fading
            K = 10**(self.K_dB / 10)
            LOS = np.sqrt(K / (K + 1))
            NLOS_real = self.np_random.normal(0, 1)
            NLOS_imag = self.np_random.normal(0, 1)
            NLOS = (NLOS_real + 1j * NLOS_imag) / np.sqrt(2)
            return LOS + NLOS
