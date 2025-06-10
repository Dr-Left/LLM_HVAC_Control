import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd

from methods.utils.plot_utils import plot_building_simulation


def save_results(env, args, reward_list, reward_breakdown_list, Parameter):
    """Save simulation results with improved data storage for flexible future plotting"""
    action_array = np.array(env.actionlist) / env.maxpower
    state_array = np.array(env.statelist)
    time_x = np.arange(len(reward_list)) * args.time_reso / 3600
    _time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create unique identifier for this simulation run
    if args.model == "ppo":
        prefix = "PPO"
    elif args.model == "mpc":
        prefix = "MPC"
    else:
        prefix = "LLM"
    sim_id = f"{prefix}_{args.build_type}_{args.climate_zone}_{args.city}_{args.model}_{args.noise:.1f}_{_time}"

    # Create results directories if they don't exist
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Save comprehensive data as a structured HDF5 file

    with h5py.File(f"results/data/{sim_id}.h5", "w") as hf:
        # Store metadata as attributes
        hf.attrs["build_type"] = args.build_type
        hf.attrs["climate_zone"] = args.climate_zone
        hf.attrs["city"] = args.city
        hf.attrs["model"] = args.model
        hf.attrs["timestamp"] = _time
        hf.attrs["time_resolution"] = args.time_reso

        # Create groups to organize data
        time_group = hf.create_group("time")
        temps_group = hf.create_group("temperatures")
        actions_group = hf.create_group("actions")
        rewards_group = hf.create_group("rewards")
        settings_group = hf.create_group("settings")

        # Store time data
        time_group.create_dataset("hours", data=time_x)

        # Store temperature data
        temps_group.create_dataset(
            "room_temperatures", data=state_array[:, : env.roomnum]
        )
        temps_group.create_dataset(
            "outside_temperature", data=state_array[:, env.roomnum]
        )
        temps_group.create_dataset("target_temperature", data=env.target[0])

        # Store action data
        actions_group.create_dataset("valve_openings", data=action_array)

        # Store reward data
        rewards_group.create_dataset("total_rewards", data=reward_list)

        # # Store reward breakdown components
        # for i in range(len(reward_breakdown_list[0])):
        #     reward_components = [breakdown[i] for breakdown in reward_breakdown_list]
        #     rewards_group.create_dataset(f'component_{i}', data=reward_components)

        # Store building parameters
        for i, building in enumerate(Parameter["buildall"]):
            building_group = settings_group.create_group(f"building_{i}")
            building_group.attrs["name"] = building[0]
            # You can add more building-specific parameters here if needed

    # 2. Also save as CSV for backward compatibility and easy inspection
    df = pd.DataFrame(reward_breakdown_list)
    df["reward"] = reward_list
    df["actions"] = [str(action[: env.roomnum]) for action in action_array]
    df["states"] = [str(state[: env.roomnum]) for state in state_array]
    df["target"] = env.target[0]
    df["OutTemp"] = env.OutTemp[: env.epochs]
    df["hours"] = time_x
    df.to_csv(f"results/data/{sim_id}.csv", index=False)

    # 3. Create both original plots but save plot data and configuration
    plot_data = (env, args, time_x, state_array, action_array, reward_list, Parameter)
    plot_building_simulation(
        data=plot_data, is_env_data=True, sim_id=sim_id, save_plots=True
    )

    return sim_id  # Return the simulation ID for reference


def list_simulations():
    """List all available simulations with useful metadata"""
    import glob

    simulation_files = glob.glob("results/data/*.h5")
    simulations = []

    for file_path in simulation_files:
        with h5py.File(file_path, "r") as hf:
            sim_info = {
                "id": os.path.basename(file_path).replace(".h5", ""),
                "build_type": hf.attrs.get("build_type", "unknown"),
                "climate_zone": hf.attrs.get("climate_zone", "unknown"),
                "city": hf.attrs.get("city", "unknown"),
                "model": hf.attrs.get("model", "unknown"),
                "timestamp": hf.attrs.get("timestamp", "unknown"),
                "file_path": file_path,
            }
            simulations.append(sim_info)

    # Convert to DataFrame for easy viewing
    return pd.DataFrame(simulations)


def load_simulation(sim_id):
    """Load a simulation by ID or file path for further analysis"""

    # Determine the file path
    if sim_id.endswith(".h5"):
        file_path = sim_id
    else:
        file_path = f"results/data/{sim_id}.h5"

    # Load data into a more convenient structure
    with h5py.File(file_path, "r") as hf:
        data = {
            # Metadata
            "metadata": {k: hf.attrs[k] for k in hf.attrs.keys()},
            # Time data
            "time": {"hours": hf["time"]["hours"][:]},
            # Temperature data
            "temperatures": {
                "rooms": hf["temperatures"]["room_temperatures"][:],
                "outside": hf["temperatures"]["outside_temperature"][:],
                "target": hf["temperatures"]["target_temperature"][()],
            },
            # Action data
            "actions": {"valve_openings": hf["actions"]["valve_openings"][:]},
            # Reward data
            "rewards": {"total": hf["rewards"]["total_rewards"][:]},
            # Building settings
            "settings": {},
        }

        # Load reward components
        for component_name in hf["rewards"].keys():
            if component_name != "total_rewards":
                data["rewards"][component_name] = hf["rewards"][component_name][:]

        # Load building settings
        for building_key in hf["settings"].keys():
            building_group = hf["settings"][building_key]
            building_name = building_group.attrs.get("name", building_key)
            data["settings"][building_name] = {
                k: building_group.attrs[k] for k in building_group.attrs.keys()
            }

    return data


# Example usage of the new functions (can be used in a separate script)
"""
# List all available simulations
simulations = list_simulations()
print(simulations)

# Load a specific simulation by ID
sim_id = simulations.iloc[0]['id']  # Get the first simulation
sim_data = load_simulation(sim_id)

# Create custom plots with different styles and limits
figs = plot_simulation(
    sim_data, 
    custom_style='ggplot',
    y_limits={'temperatures': (18, 30), 'actions': (0, 1), 'rewards': (-10, 10)},
    figsize=(16, 12),
    colors=['red', 'blue', 'green', 'purple']  # Custom colors for rooms
)

# Further customize plots if needed
figs['main'].axes[0].set_title('Custom Temperature Plot')
figs['performance'].axes[0].grid(False)

"""
