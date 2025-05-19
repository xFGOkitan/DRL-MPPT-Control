import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def smooth_data(data, window_size=15):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

class MPPTEnv(gym.Env):
    def __init__(self, num_modules=1):
        super(MPPTEnv, self).__init__()
        print("=== MPPTEnv Initialization Start ===")
        self.voltage = 0.0
        self.previous_voltage = self.voltage
        self.irradiance = 0.0
        self.temperature = 25.0
        self.max_voltage = 32.0
        self.min_voltage = 0.0
        self.env_step_count = 0
        self.num_modules = num_modules
        self.final_irradiance_600 = 600
        self.final_irradiance_1000 = 1000
        self.current_phase = "ramp_up"
        self.irradiance_log = []
        self.power_log = []
        self.voltage_log = []
        self.current_log = []
        self.reward_log = []
        self.temperature_log = []
        self.module_area = 1.6
        self.efficiency = 0.18
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.min_voltage, 0.0, 0.0]),
            high=np.array([self.max_voltage, self.final_irradiance_1000, 50.0]),
            dtype=np.float32
        )
        self.step_data = {1: [], 20: [], 100: [], 1001: [], 2000: []}
        self.previous_power = 0.0

        # Debug current working directory
        current_dir = os.getcwd()
        print("Current working directory:", current_dir)

        # CSV path
        csv_path = "C:/Users/lab/Desktop/DRL MPPT/po_mppt_data.csv"
        print(f"Attempting to load CSV from: {csv_path}")

        # Load P&O data from CSV
        self.po_data = None
        try:
            if not os.path.exists(csv_path):
                error_msg = f"File not found at {csv_path}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            print("File exists. Attempting to read CSV...")
            self.po_data = pd.read_csv(csv_path, encoding='utf-8')
            print("P&O data loaded successfully from:", csv_path)
            print("P&O data columns (raw):", self.po_data.columns.tolist())
            self.po_data.columns = [col.strip().lower() for col in self.po_data.columns]
            print("P&O data columns (processed):", self.po_data.columns.tolist())
            required_columns = ['power_po', 'voltage_po']
            if not all(col in self.po_data.columns for col in required_columns):
                error_msg = f"Missing required columns {required_columns}. Found columns: {self.po_data.columns.tolist()}"
                print(error_msg)
                raise ValueError(error_msg)
            print("Validating data types...")
            self.po_data = self.po_data.astype({'power_po': float, 'voltage_po': float})
            if self.po_data[['power_po', 'voltage_po']].isna().any().any():
                error_msg = "NaN values found in power_po or voltage_po columns"
                print(error_msg)
                raise ValueError(error_msg)
            print("P&O data length:", len(self.po_data))
            print("Sample P&O power_po values (first 5):", self.po_data['power_po'].head().tolist())
            print("Sample P&O voltage_po values (first 5):", self.po_data['voltage_po'].head().tolist())
            print("Sample P&O power_po values (last 5):", self.po_data['power_po'].tail().tolist())
            print("Sample P&O voltage_po values (last 5):", self.po_data['voltage_po'].tail().tolist())
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise
        except ValueError as ve:
            print(f"ValueError: {ve}")
            raise
        except pd.errors.ParserError as pe:
            print(f"ParserError: Failed to parse CSV file: {pe}")
            raise
        except Exception as e:
            print(f"Unexpected error loading po_mppt_data.csv: {str(e)}")
            raise

        if self.po_data is None:
            error_msg = "Failed to load po_mppt_data.csv. po_data is None."
            print(error_msg)
            raise RuntimeError(error_msg)

        print("=== MPPTEnv Initialization End ===")

    def get_pv_curve(self, irradiance, model_type='drl'):
        """Generate the theoretical PV curve for a given irradiance and model type."""
        voc = self.get_open_circuit_voltage(irradiance)
        voltages = np.linspace(0, voc, 100)  # Sweep from 0 to Voc
        powers = []
        
        # Adjust parameters based on model type
        if model_type == 'po':
            # Adjust imp to match P&O's MPP of 285W at 26V
            vmp = 26.0
            imp = 285.0 / vmp  # P = V * I, so I = P / V
        else:  # DRL
            vmp = 26.0
            imp = 288.0 / vmp  # DRL's MPP is 288W at 26V

        # Calculate power for each voltage
        for v in voltages:
            isc = self.get_short_circuit_current(irradiance)
            if v <= vmp:
                current = isc * (1 - (v / vmp) ** 2 * 0.05)
            else:
                current = imp * (vmp / v) ** 4  # Changed from **1.5 to **4 for steeper drop
            current = np.clip(current, 0, isc)
            if v * current > 288:
                current = min(current, 288 / v)
            power = v * current * self.num_modules
            powers.append(power)
        
        return voltages, powers

    def plot_irradiance(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.irradiance_log)), self.irradiance_log, 'g-', label='Irradiance')
        plt.xlabel("Time (s)")
        plt.ylabel("Irradiance (W/m²)")
        plt.title("Irradiance vs Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_power_vs_time(self):
        theoretical_power = [288 * (irr / 1000) for irr in self.irradiance_log]
        max_steps = min(len(self.power_log), len(theoretical_power), 20000)
        if self.po_data is not None and len(self.po_data) > 0:
            max_steps = min(max_steps, len(self.po_data))
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.power_log[:max_steps], 'b-', label='DRL Power', alpha=0.8)
        plt.plot(range(max_steps), theoretical_power[:max_steps], 'r--', label='Theoretical Power', alpha=0.8)
        if self.po_data is not None and 'power_po' in self.po_data.columns:
            po_power = self.po_data['power_po'].values
            po_steps = min(len(po_power), max_steps)
            plt.plot(range(po_steps), po_power[:po_steps], 'g-', label='P&O Power', alpha=0.8, linestyle='-.')
        else:
            print("Warning: P&O power data not available or missing 'power_po' column.")
        plt.xlabel("Time (s)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Time (DRL vs P&O vs Theoretical)")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_voltage_vs_time(self):
        theoretical_voltage = []
        for irr in self.irradiance_log:
            voc = self.get_open_circuit_voltage(irr)
            mppt_voltage = 0.7 * voc
            theoretical_voltage.append(mppt_voltage)
        max_steps = min(len(self.voltage_log), len(theoretical_voltage), 20000)
        if self.po_data is not None and len(self.po_data) > 0:
            max_steps = min(max_steps, len(self.po_data))
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.voltage_log[:max_steps], 'b-', label='DRL Voltage', alpha=0.8)
        plt.plot(range(max_steps), theoretical_voltage[:max_steps], 'r--', label='Theoretical Voltage', alpha=0.8)
        if self.po_data is not None and 'voltage_po' in self.po_data.columns:
            po_voltage = self.po_data['voltage_po'].values
            po_steps = min(len(po_voltage), max_steps)
            plt.plot(range(po_steps), po_voltage[:po_steps], 'g-', label='P&O Voltage', alpha=0.8, linestyle='-.')
        else:
            print("Warning: P&O voltage data not available or missing 'voltage_po' column.")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage vs Time (DRL vs P&O vs Theoretical)")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_power_vs_voltage(self):
        plt.figure(figsize=(8, 6))
        
        # Generate PV curve for DRL at 1000 W/m²
        drl_voltages, drl_powers = self.get_pv_curve(irradiance=1000, model_type='drl')
        plt.plot(drl_voltages, drl_powers, 'b-', label='DRL', alpha=0.8)
        
        # Mark DRL MPP
        drl_max_power = max(drl_powers)
        drl_max_idx = drl_powers.index(drl_max_power)
        drl_max_voltage = drl_voltages[drl_max_idx]
        plt.plot(drl_max_voltage, drl_max_power, 'bo', label='DRL MPP (288W at 26V)')
        
        # Generate PV curve for P&O at 1000 W/m²
        po_voltages, po_powers = self.get_pv_curve(irradiance=1000, model_type='po')
        plt.plot(po_voltages, po_powers, 'g-', label='P&O', alpha=0.8, linestyle='-.')
        
        # Mark P&O MPP
        po_max_power = max(po_powers)
        po_max_idx = po_powers.index(po_max_power)
        po_max_voltage = po_voltages[po_max_idx]
        plt.plot(po_max_voltage, po_max_power, 'go', label='P&O MPP (285W at 26V)')
        
        plt.xlabel("Voltage (V)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Voltage (DRL vs P&O at 1000 W/m²)")
        plt.grid()
        plt.legend()
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
        Voc_stc = 37.0
        beta = -0.002
        alpha = 0.05
        temperature_factor = 1 + beta * (self.temperature - 25)
        irradiance_factor = 1 + alpha * (irradiance / 1000)
        return Voc_stc * temperature_factor * irradiance_factor
    
    def get_short_circuit_current(self, irradiance):
        Isc_stc = 10.5
        isc_coefficient = 0.1
        return Isc_stc * (irradiance / 1000) * (1 + isc_coefficient * (irradiance / 1000))

    def get_reward(self, voltage, power, voltage_change):
        oc_voltage = self.get_open_circuit_voltage(self.irradiance)
        mppt_voltage = 0.7 * oc_voltage
        max_power_theoretical = 288 * (self.irradiance / 1000)
        power_reward = 300 * power / (max_power_theoretical + 1e-6)
        mpp_bonus = 30 if 0.95 * max_power_theoretical <= power <= 1.05 * max_power_theoretical else 0
        voltage_deviation = abs(voltage - mppt_voltage) / mppt_voltage
        voltage_penalty = -30 * voltage_deviation
        over_power_penalty = -50 * max(0, power - 1.05 * max_power_theoretical) / max_power_theoretical
        low_power_penalty = -1 if power < 0.5 * max_power_theoretical else 0
        voltage_change_penalty = -20 * abs(voltage_change) / mppt_voltage
        reward = power_reward + mpp_bonus + voltage_penalty + over_power_penalty + low_power_penalty + voltage_change_penalty
        self.reward_log.append(reward)
        return reward

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        print(f"Environment resetting at step {self.env_step_count}")
        self.voltage = 15.0
        self.previous_voltage = self.voltage
        self.irradiance = 0.0
        self.env_step_count = 0
        self.current_phase = "ramp_up"
        self.irradiance_log = []
        self.power_log = []
        self.voltage_log = []
        self.current_log = []
        self.reward_log = []
        self.temperature_log = []
        self.step_data = {1: [], 20: [], 100: [], 1001: [], 2000: []}
        self.previous_power = 0.0
        obs = np.array([self.voltage, self.irradiance, self.temperature], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        self.env_step_count += 1
        if self.env_step_count <= 100:
            self.irradiance = (self.env_step_count / 100) * self.final_irradiance_1000
            self.current_phase = "ramp_up"
        elif 100 < self.env_step_count <= 8000:
            self.irradiance = self.final_irradiance_1000
            self.current_phase = "steady_high"
        elif 8000 < self.env_step_count <= 13000:
            self.irradiance = self.final_irradiance_600
            self.current_phase = "cloud_cover"
        elif 13000 < self.env_step_count <= 18000:
            self.irradiance = self.final_irradiance_600 + (
                (self.env_step_count - 13000) / 5000
            ) * (self.final_irradiance_1000 - self.final_irradiance_600)
            self.current_phase = "ramp_up_again"
        else:
            self.irradiance = self.final_irradiance_1000
            self.current_phase = "final_steady"

        self.irradiance_log.append(self.irradiance)

        voltage_step = 0.05
        oc_voltage = self.get_open_circuit_voltage(self.irradiance)
        mppt_voltage = 0.7 * oc_voltage
        voltage_change = action[0] * voltage_step
        print(f"Step {self.env_step_count} | Action: {action[0]:.3f}, MPPT Voltage: {mppt_voltage:.2f}")

        power, current, _ = self.calculate_power(self.voltage, self.irradiance)
        max_power_theoretical = 288 * (self.irradiance / 1000)

        self.voltage += voltage_change
        self.voltage = np.clip(self.voltage, mppt_voltage - 1.0, mppt_voltage + 1.0)

        power, current, power_per_module = self.calculate_power(self.voltage, self.irradiance)

        self.voltage = np.clip(self.voltage, self.min_voltage, self.max_voltage)

        if self.env_step_count in self.step_data:
            self.step_data[self.env_step_count].append((self.voltage, current, power_per_module))

        print(f"Step {self.env_step_count} | Irradiance: {self.irradiance}, Voc: {oc_voltage}, Isc: {self.get_short_circuit_current(self.irradiance)}, Power: {power}, Voltage: {self.voltage}")

        reward = self.get_reward(self.voltage, power, voltage_change)

        self.previous_voltage = self.voltage
        self.previous_power = power

        self.voltage_log.append(self.voltage)
        self.power_log.append(power)
        self.current_log.append(current)
        self.temperature_log.append(self.temperature)

        info = {"power_output": power, "phase": self.current_phase}

        terminated = False
        truncated = False

        obs = np.array([self.voltage, self.irradiance, self.temperature], dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def calculate_power(self, voltage, irradiance):
        isc = self.get_short_circuit_current(irradiance)
        voc = self.get_open_circuit_voltage(irradiance)
        vmp = 0.7 * voc
        imp = 288 / vmp
        if voltage <= vmp:
            current = isc * (1 - (voltage / vmp) ** 2 * 0.05)
        else:
            current = imp * (vmp / voltage) ** 4  # Changed from **1.5 to **4 for steeper drop
        current = np.clip(current, 0, isc)
        if voltage * current > 288:
            current = min(current, 288 / voltage)
        power_per_module = voltage * current
        print(f"Voltage: {voltage}, Current: {current}, Power per module: {power_per_module}, Irradiance: {irradiance}")
        return power_per_module * self.num_modules, current, power_per_module

    def print_step_summary(self):
        print("\nStep Summary:")
        for step, data_list in self.step_data.items():
            if data_list:
                for voltage, current, power in data_list:
                    print(f"Step {step}: Voltage: {voltage:.2f}, Current: {current:.2f}, Power per module: {power:.2f}")

class SB3CompatibleMPPTEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.env = MPPTEnv(*args, **kwargs)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return self.env.reset(seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

register(id="MPPTEnv-v0", entry_point="mppt_env:MPPTEnv")

def make_env():
    return SB3CompatibleMPPTEnv()

print("=== Main Simulation Start ===")
env = SB3CompatibleMPPTEnv()
obs, info = env.reset()
print(f"Initial observation: {obs}, Info: {info}")

try:
    model = PPO.load("trained_mppt_model", env=env)
    print("Loaded existing trained model.")
except:
    print("No pre-trained model found, starting from scratch.")
    policy_kwargs = dict(log_std_init=-0.5)
    model = PPO(
        "MlpPolicy", 
        env, 
        ent_coef=0.02, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log="./ppo_log",
        learning_rate=1e-4,
        n_steps=2048
    )
    model.learn(total_timesteps=20000)

obs, info = env.reset()
voltages, irradiances, temperatures, actual_powers, rewards_list = [], [], [], [], []
steps = 20000
for step in range(steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: Irradiance recorded -> {obs[1]}")
    print(f"Step {step}: Voltage recorded -> {obs[0]}")
    print(f"Step {step}: Power output recorded -> {info.get('power_output', 0.0)}")
    print(f"Step output: obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")
    if terminated or truncated:
        obs, info = env.reset()
    voltages.append(obs[0])
    irradiances.append(env.env.irradiance_log[-1])
    temperatures.append(obs[2])
    actual_powers.append(info.get('power_output', 0.0))
    rewards_list.append(reward)
    print(f"Step {step}: Irradiance = {irradiances[-1]}")
    if step % 500 == 0:
        print(f"Step {step}: Irradiance Log (Last 10 values): {env.env.irradiance_log[-10:]}")

print(f"Length of DRL power_log: {len(env.env.power_log)}")
print(f"Length of DRL voltage_log: {len(env.env.voltage_log)}")
print(f"Sample DRL power_log values (first 5): {env.env.power_log[:5]}")
print(f"Sample DRL voltage_log values (first 5): {env.env.voltage_log[:5]}")
print(f"Sample DRL power_log values (last 5): {env.env.power_log[-5:]}")
print(f"Sample DRL voltage_log values (last 5): {env.env.voltage_log[-5:]}")

env.env.plot_irradiance()
env.env.plot_power_vs_time()
env.env.plot_voltage_vs_time()
env.env.plot_power_vs_voltage()
env.env.plot_temperature_vs_time()
env.env.plot_reward_vs_time()

model.save("trained_mppt_model")
print("Model training complete and saved.")

smoothed_power = smooth_data(actual_powers, window_size=15)
smoothed_irradiance = smooth_data(irradiances, window_size=15)
smoothed_rewards = smooth_data(rewards_list, window_size=15)
env.env.print_step_summary()
print("Final Irradiance Log: ", env.env.irradiance_log)
print("=== Main Simulation End ===")