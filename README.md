# Reinforcement-Learning-Assisted-Pilot-Scheduling-and-Pilot-Power-Assignment-in-Urban-5G-Environment

This project simulates an adaptive pilot transmission mechanism for wireless communication systems operating in time-varying channels, such as those encountered in mobile environments.

# Objective
The simulation aims to minimize the channel estimation error (NMSE) while efficiently utilizing pilot signals. Instead of transmitting pilots at every interval, the system dynamically decides when to send a pilot and adjusts the pilot power based on the current NMSE.

# Reward Function
We define the reward at each timestep as 
reward = throughput - alpha * pilot_overhead - beta * pilot_power_cost - gamma * nmse_penalty

Throughput: System throughput achieved at time.

PilotOverhead: Cost due to time or bandwidth consumed by pilot signals.

PilotPowerCost: Energy or power consumed to transmit pilot signals.

NMSEPenalty: Normalized Mean Squared Error; higher error leads to a greater penalty.

α,β,γ: Tunable coefficients to balance the trade-offs between throughput, overhead, power, and estimation accuracy
# Key Features
o Adaptive Pilot Scheduling: Pilots are transmitted only when needed, reducing unnecessary overhead.

o Pilot Power Control: Dynamically varies the pilot power depending on the channel condition.

o NMSE Tracking: Continuously monitors the normalized mean square error for estimation accuracy.

o Mobility Modeling: Incorporates Doppler effects to simulate real-world mobility scenarios.

This Simulation Helps Evaluate
o How frequently pilots should be sent in mobile environments.

o The trade-off between pilot power and estimation accuracy.

o The impact of Doppler shifts (user mobility) on system performance.

# Potential Research Applications
5G/6G channel estimation and pilot design.

Mobility-aware optimization in wireless networks.

Reinforcement Learning (RL)-based adaptive pilot scheduling.

Energy vs. accuracy trade-offs in Massive MIMO systems.

.
