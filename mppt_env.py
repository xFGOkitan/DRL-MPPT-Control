import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

# Simple moving average function for smoothing
def smooth_data(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

class MPPTEnv(gym.Env):
    def __init__(self, num_modules=1):
        super(MPPTEnv, self).__init__()

        self.voltage = 10.0
        self.irradiance = 0.0  # Initial irradiance
        self.temperature = 25.0
        self.max_voltage = 20.0
        self.min_voltage = 0.0

        self.env_step_count = 0
        self.num_modules = num_modules
        self.ramp_up_rate = 0.1  # Adjust the ramp rate as needed
        self.ramp_down_rate = 0.05  # Adjust the ramp rate as needed
        self.final_irradiance_600 = 600  # 600 W/m² (target value)
        self.final_irradiance_1000 = 1000  # 1000 W/m² (target value)
        self.steady_high_duration = 5000  # Duration at 600 W/m²
        self.ramp_down_duration = 8000  # Duration to ramp down from 1000 W/m² to 600 W/m²
        self.current_phase = "ramp_up"  # Initial phase: ramp-up from 0 to 1000 W/m²
        self.irradiance = 0  # Starting irradiance value
        self.phase_start_time = 0  # Time step when each phase starts
        self.steady_high_step = 0  # Step at which steady high phase starts
        self.irradiance_log = []  # Store irradiance values over time
        self.power_log = []  # To store power for plotting
        self.voltage_log = []  # To store voltage for plotting
        self.reward_log = []  # To store reward for plotting
        self.temperature_log = []  #To store temp for plotting

        # Power calculation constants
        self.module_area = 1.6  # m²
        self.efficiency = 0.18

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Actions: 0 = Decrease, 1 = Maintain, 2 = Increase
        self.observation_space = spaces.Box(
            low=np.array([self.min_voltage, 0.0, 0.0]),
            high=np.array([self.max_voltage, self.final_irradiance_1000, 50.0]),
            dtype=np.float32
        )
        
    def plot_irradiance(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.irradiance_log)), self.irradiance_log)
        plt.xlabel("Time (s)")
        plt.ylabel("Irradiance (W/m²)")
        plt.title("Irradiance vs Time")
        plt.grid(True)
        plt.show()

    def plot_power_vs_time(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.power_log)), self.power_log)
        plt.xlabel("Time (s)")
        plt.ylabel("Power Output (W)")
        plt.title("Raw Power Output vs Time")
        plt.grid()
        plt.show()

    def plot_voltage_vs_time(self):
        print(f"Plotting Voltage: Total points = {len(self.voltage_log)}")
        print(f"Sample voltage values: {self.voltage_log[-5:]}")
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.voltage_log)), self.voltage_log)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Raw Voltage vs Time")
        plt.grid()
        plt.show()

    def plot_power_vs_voltage(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.voltage_log, self.power_log, s=1, alpha=0.5)
        plt.xlabel("Voltage (V)")
        plt.ylabel("Power Output (W)")
        plt.title("Raw Power Output vs Voltage")
        plt.grid()
        plt.show()

    def plot_temperature_vs_time(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.temperature_log)), self.temperature_log)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Change vs Time")
        plt.grid()
        plt.show()

    def plot_reward_vs_time(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.reward_log)), self.reward_log)
        plt.xlabel("Time (s)")
        plt.ylabel("Reward")
        plt.title("Reward vs Time")
        plt.grid()
        plt.show()


    def get_open_circuit_voltage(self, irradiance):
        Voc_stc = 21.0
        beta = -0.002
        alpha = 0.05
        temperature_factor = 1 + beta * (self.temperature - 25)
        irradiance_factor = 1 + alpha * (self.irradiance / 1000)
        return Voc_stc * temperature_factor * irradiance_factor
    
    def get_short_circuit_current(self, irradiance):
        Isc_stc = 8.0  # Short-circuit current at standard test conditions
        isc_coefficient = 0.1  # Coefficient for short-circuit current change with irradiance
        return Isc_stc * (1 + isc_coefficient * (irradiance / 1000))

    def get_reward(self, voltage, power, action):
        oc_voltage = self.get_open_circuit_voltage(self.irradiance)
        mppt_voltage = 0.8 * oc_voltage
        voltage_factor = max(0, 1 - (abs(voltage - mppt_voltage) / mppt_voltage))

        # Dynamically calculate max_power based on current irradiance and Isc
        isc = self.get_short_circuit_current(self.irradiance)
        max_power = isc * self.get_open_circuit_voltage(self.irradiance) * self.module_area * self.efficiency
    
        power_factor = power / (max_power + 1e-6)
        reward = abs(voltage - mppt_voltage)

        if action == 1:
            reward -= 0.2
        if voltage < self.min_voltage or voltage > self.max_voltage:
            reward -= 0.5

        # Log reward for tracking (optional, for debugging)
        self.reward_log.append(reward)  # Save reward for visualization

        return reward

    def reset(self, seed=None, **kwargs):
        # Handle seeding for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize the environment state
        self.voltage = 10.0
        self.irradiance = 0.0
        self.env_step_count = 0
        self.current_phase = "ramp_up"

        # Return the initial state as observation and a blank info dictionary
        obs = np.array([self.voltage, self.irradiance, self.temperature], dtype=np.float32)
        info = {}  # Return an empty dictionary for now
        return obs, info  # Return both observation and info

    def step(self, action):
        # Increment the step count
        self.env_step_count += 1

        # Phase 1: Ramp-up from 0 to 1000 W/m² in the first 100 steps
        if self.env_step_count <= 100:
            self.irradiance = (self.env_step_count / 100) * self.final_irradiance_1000
            self.current_phase = "ramp_up"
    
        # Phase 2: Hold at 1000 W/m² until step 8000
        elif 100 < self.env_step_count <= 8000:
            self.irradiance = self.final_irradiance_1000
            self.current_phase = "steady_high"

        # Phase 3: Drop to 600 W/m² at step 8000 and hold for 5000 steps
        elif 8000 < self.env_step_count <= 13000:
            self.irradiance = self.final_irradiance_600
            self.current_phase = "cloud_cover"

        # Phase 4: Ramp-up back to 1000 W/m² from step 13000 to 18000
        elif 13000 < self.env_step_count <= 18000:
            self.irradiance = self.final_irradiance_600 + (
                (self.env_step_count - 13000) / 5000
            ) * (self.final_irradiance_1000 - self.final_irradiance_600)
            self.current_phase = "ramp_up_again"

        # Phase 5: Hold at 1000 W/m² for the rest of the simulation
        else:
            self.irradiance = self.final_irradiance_1000
            self.current_phase = "final_steady"

        # Log irradiance data
        self.irradiance_log.append(self.irradiance)

        # Calculate voltage based on irradiance
        voltage = self.get_open_circuit_voltage(self.irradiance)  # Update voltage based on irradiance
        power = self.calculate_power(voltage, self.irradiance)  # Calculate power

        # Log the values for further debugging
        print(f"Step {self.env_step_count} | Power: {power}, Voltage: {voltage}, Irradiance: {self.irradiance}")

        # Assign reward value (ensure reward is always set)
        reward = 0  # Default reward if conditions don't meet
        if action == 'some_condition':  # Check condition for reward (replace with actual logic)
            reward = self.get_reward(self.voltage, power, action)  # Calculate actual reward

        # Log data for plotting
        self.voltage_log.append(voltage)  # Log the voltage value
        self.power_log.append(power)  # Log the power value
        self.temperature_log.append(self.temperature)  # Log temperature
        

        # Power calculation
        power_per_module = self.irradiance * self.module_area * self.efficiency
        total_power = power_per_module * self.num_modules
        print(f"Step {self.env_step_count}: Power per module = {power_per_module} W")
        print(f"Step {self.env_step_count}: Total power = {total_power} W")

        # Reward calculation
        reward = self.get_reward(self.voltage, total_power, action)

        # Termination conditions
        terminated = self.voltage <= self.min_voltage or self.voltage >= self.max_voltage
        truncated = False

        # Observation and info
        obs = np.array([self.voltage, self.irradiance, self.temperature], dtype=np.float32)
        info = {"power_output": total_power, "phase": self.current_phase}

        return obs, reward, terminated, truncated, info

    def calculate_power(self, voltage, irradiance):
        # Recalculate power for each module with updated voltage and irradiance values
        power_per_module = self.irradiance * self.module_area * self.efficiency
        return power_per_module * self.num_modules



# Wrapper to ensure compatibility with Stable-Baselines3
class SB3CompatibleMPPTEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        # Initialize the environment
        self.env = MPPTEnv(*args, **kwargs)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed=None):
        # Seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize environment state with initial values
        self.voltage = 10.0
        self.irradiance = 0.0
        self.temperature = 25.0
        self.env_step_count = 0
        self.current_phase = "ramp_up"
        self.irradiance_log = []  # Reset irradiance log on new episodes

        # Return the initial observation
        obs = np.array([self.voltage, self.irradiance, self.temperature], dtype=np.float32)
        info = {}  # You can add any extra information here if needed
        return obs, info  # Return only two values: obs and info

    def step(self, action):
        # Unpack the 4 returned values from the step method
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Return the expected 5 values: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)


# Register the environment
register(
    id="MPPTEnv-v0",
    entry_point="__main__:MPPTEnv",
)

# Wrap the environment for compatibility with Stable-Baselines3
def make_env():
    return SB3CompatibleMPPTEnv()

env = SB3CompatibleMPPTEnv()

# Ensure env reset works correctly
obs, info = env.reset()
print(f"Initial observation: {obs}, Info: {info}")


# Load or initialize the PPO model (finding trained data file/creating new training if not found)
try:
    model = PPO.load("trained_mppt_model", env=env)
    print("Loaded existing trained model.")
except:
    print("No pre-trained model found, starting from scratch.")
    model = PPO("MlpPolicy", env, learning_rate=0.0001, n_steps=2048, verbose=1)

# Modify policy_kwargs for wider exploration
policy_kwargs = dict(log_std_init=-0.5)  # Option 2: higher exploration at start

# Create PPO model with modified ent_coef and policy_kwargs for better exploration
model = PPO(
    "MlpPolicy",
    env,
    ent_coef=0.02,                # Option 1: encourages more exploration
    policy_kwargs=policy_kwargs, # Apply custom policy with wider initial std
    verbose=1,
    tensorboard_log="./ppo_log"
)

# Train the model
model.learn(total_timesteps=20000)

# After training, plot the irradiance using the original environment
env.env.plot_irradiance()  # This will call the plot_irradiance() from the MPPTEnv class
env.env.plot_power_vs_time()
env.env.plot_voltage_vs_time()
env.env.plot_power_vs_voltage()
env.env.plot_temperature_vs_time()
env.env.plot_reward_vs_time()

# Save the trained model
model.save("trained_mppt_model")
print("Model training complete and saved.")

# Data collection for plotting
obs, info = env.reset()  # Reset the environment before collecting data
voltages, irradiances, temperatures, actual_powers, rewards_list = [], [], [], [], []

steps = 20000  # Total steps you want to train for
for step in range(steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Debugging: Print recorded values  
    print(f"Step {step}: Irradiance recorded -> {obs[1]}")
    print(f"Step {step}: Voltage recorded -> {obs[0]}")
    print(f"Step {step}: Power output recorded -> {info.get('power_output', 0.0)}")

    # Print what is returned
    print(f"Step output: obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")
    
    if terminated or truncated:
        obs, info = env.reset()

    # Collecting data for plotting
    voltages.append(obs[0])        # Voltage
    print(f"Recorded Irradiance: {obs[1]}")  # Just before appending to the list
    irradiances.append(env.env.irradiance_log[-1])     # Irradiance
    temperatures.append(obs[2])    # Temperature
    actual_powers.append(info.get('power_output', 0.0))
    rewards_list.append(reward)    # Collect reward for plotting

    # Debugging line to check irradiance values at each step
    print(f"Step {step}: Irradiance = {irradiances[-1]}")

    # Print the irradiance log every 500 steps (or any other condition)
    if step % 500 == 0:
        print(f"Step {step}: Irradiance Log (Last 10 values): {env.env.irradiance_log[-10:]}")

smoothed_power = smooth_data(actual_powers, window_size=10)
smoothed_irradiance = smooth_data(irradiances, window_size=10)
smoothed_rewards = smooth_data(rewards_list, window_size=10)

# After all episodes, print the final irradiance log
print("Final Irradiance Log: ", env.env.irradiance_log)