import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods.utils import load_env, reward_func
from methods.utils.plot_utils import plot_building_simulation
from methods.utils.save_results import list_simulations, load_simulation


def plot_ppo_rewards(sim_id):
    """Plot the training rewards for PPO simulations."""
    rewards_file = os.path.join("rewards", f"{sim_id.split('_ppo_')[0]}.csv")
    if not os.path.exists(rewards_file):
        print(f"Rewards file not found: {rewards_file}")
        return

    rewards_data = pd.read_csv(rewards_file)
    rewards = rewards_data.iloc[
        :, 1
    ].values  # Assuming rewards are in the second column

    # Create the plot
    plt.figure(figsize=(10, 6))
    # font
    plt.rcParams["font.family"] = "Songti SC"
    plt.rcParams["font.size"] = 24
    plt.plot(rewards)
    # plt.title(f"PPO训练奖励")
    plt.xlabel("训练步数")
    plt.ylabel("奖励")
    plt.style.use("seaborn-v0_8")
    plt.tight_layout()
    plt.grid(True)

    # Save the plot
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(f"results/figures/{sim_id}_training_reward.png")
    plt.close()
    print(f"figure saved to results/figures/{sim_id}_training_reward.png")

    # Print statistics
    print("Training Rewards Statistics:")
    print("Mean reward:", np.mean(rewards))
    print("Std reward:", np.std(rewards))


def main(sim_id):
    # For PPO simulations, plot the training rewards
    if sim_id.startswith("PPO"):
        plot_ppo_rewards(sim_id)

    parts = sim_id.split("_")
    params = {
        "Building": parts[1],  # OfficeMedium
        "Weather": "_".join(parts[2:4]),  # Cool_Humid
        "Location": parts[4],  # Buffalo
        "time_reso": 360,  # Default time resolution
    }

    # Load environment with extracted parameters
    env, Parameter = load_env(params, reward_func)
    sim_data = load_simulation(sim_id)
    # Create custom plots with different styles and limits
    figs = plot_building_simulation(
        data=sim_data,
        is_env_data=False,
        # custom_style='ggplot',
        # y_limits={'temperatures': (18, 30), 'actions': (0, 1), 'rewards': (-10, 10)},
        # figsize=(16, 12),
        sim_id=sim_id,
        save_plots=True,
    )

    # Further customize plots if needed
    figs["temperature"].axes[0].set_title("Custom Temperature Plot")
    figs["action"].axes[0].set_title("Custom Action Plot")
    figs["performance"].axes[0].grid(False)

    # Print logs similar to llm.py lines 123-141
    print("################PLOT ANALYSIS is Done############")

    # Extract data for statistics
    rewards = sim_data["rewards"]["total"]
    print("reward_mean: ", np.mean(rewards))
    print("reward_std: ", np.std(rewards))

    # Calculate temperature statistics
    room_temps = sim_data["temperatures"]["rooms"]
    target_temp = sim_data["temperatures"]["target"]
    T = len(rewards)
    N = room_temps.shape[1]  # Number of rooms

    # Calculate average temperature deviation
    delta_t = room_temps - target_temp
    print(
        "Average standard deviation of temperature: {:.2f} C".format(
            np.sqrt(1 / T * 1 / N * np.sum(delta_t**2))
        )
    )

    # Calculate HVAC statistics
    actions = sim_data["actions"]["valve_openings"]
    # Approximating the maxpower calculation since we don't have env.maxpower directly
    print(
        "Average HVAC output action: {:.2f}".format(
            np.mean(np.sum(actions, axis=1)),
        ),
    )
    print(
        "Average HVAC output power: {:.2f} kW".format(
            np.mean(np.sum(actions * env.maxpower, axis=1)) / 1000,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-id", type=str, default=None)
    args = parser.parse_args()

    # Extract parameters from sim_id
    if args.sim_id is not None:
        main(args.sim_id)

    # List all available simulations
    if args.sim_id is None:
        simulations = list_simulations()
        print(simulations)

        # Load a specific simulation by ID
        for sim_id in simulations["id"]:
            main(sim_id)
