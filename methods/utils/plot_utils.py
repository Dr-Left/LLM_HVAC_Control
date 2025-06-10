# Direct call of this script is shown in `plot.py`
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def save_custom_plots(figs, output_dir="results/custom_figures", prefix="custom"):
    """Save custom plot figures to specified directory"""
    os.makedirs(output_dir, exist_ok=True)

    for name, fig in figs.items():
        output_path = f"{output_dir}/{prefix}_{name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
        plt.close(fig)

    return [f"{output_dir}/{prefix}_{name}.png" for name in figs.keys()]


def plot_building_simulation(
    data,
    is_env_data=True,
    custom_style=None,
    y_limits=None,
    figsize=None,
    colors=None,
    output_dir="results/figures",
    sim_id=None,
    save_plots=True,
):
    """
    Unified plotting function that works with both environment data and loaded simulation data.

    Parameters:
    -----------
    data : dict or tuple
        Either:
        - For environment data (is_env_data=True): Tuple of (env, args, time_x, state_array, action_array, reward_list, Parameter)
        - For loaded data (is_env_data=False): The simulation data dictionary loaded using load_simulation()
    is_env_data : bool
        Flag indicating if the data is from environment (True) or loaded from file (False)
    custom_style : str, optional
        The matplotlib style to use (e.g., 'ggplot', 'seaborn-v0_8-dark')
    y_limits : dict, optional
        Custom y-axis limits for different plots, e.g. {'temperatures': (18, 28), 'actions': (0, 1)}
    figsize : tuple, optional
        Custom figure size as (width, height) for main plot and performance plot
    colors : list, optional
        List of colors to use for different rooms
    output_dir : str, optional
        Directory to save plots (if save_plots=True)
    sim_id : str, optional
        Simulation ID for file naming (required if save_plots=True and not found in data)
    save_plots : bool, optional
        Whether to save the generated plots to files

    Returns:
    --------
    figs : dict
        Dictionary of matplotlib figures for further customization
    """
    # Apply custom style if provided
    if custom_style:
        plt.style.use(custom_style)
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.size": 28,
                "axes.labelsize": 22,
                "axes.titlesize": 28,
                "xtick.labelsize": 22,
                "ytick.labelsize": 22,
                "legend.fontsize": 22,
                "lines.linewidth": 2.5,
                "grid.alpha": 0.3,
                "figure.titlesize": 32,
                "font.family": ["Arial", "Songti SC"],
            }
        )

    # Extract data based on input type
    if is_env_data:
        env, args, time_x, state_array, action_array, reward_list, Parameter = data
        build_type = args.build_type
        city = args.city
        climate_zone = args.climate_zone
        room_count = env.roomnum
        room_temps = state_array[:, :room_count]
        outside_temp = state_array[:, room_count]
        target_temp = env.target[0]
        actions = action_array
        rewards = reward_list
        room_names = (
            [build[0] for build in Parameter["buildall"]]
            if "buildall" in Parameter
            else [f"Room {i+1}" for i in range(room_count)]
        )
    else:
        build_type = data["metadata"].get("build_type", "unknown")
        city = data["metadata"].get("city", "unknown")
        climate_zone = data["metadata"].get("climate_zone", "unknown")
        time_x = data["time"]["hours"]
        room_temps = data["temperatures"]["rooms"]
        outside_temp = data["temperatures"]["outside"]
        target_temp = data["temperatures"]["target"]
        actions = data["actions"]["valve_openings"]
        rewards = data["rewards"]["total"]
        room_count = room_temps.shape[1]
        room_names = (
            list(data["settings"].keys())
            if "settings" in data and data["settings"]
            else [f"Room {i+1}" for i in range(room_count)]
        )

    # Generate room colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, room_count))

    figs = {}

    # Temperature plot
    plt.figure(figsize=figsize or (14, 6))
    for i in range(room_count):
        room_name = room_names[i] if i < len(room_names) else f"房间 {i+1}"
        plt.plot(time_x, room_temps[:, i], label=room_name, color=colors[i])
    plt.plot(
        time_x,
        outside_temp,
        label="室外温度",
        linestyle="--",
        linewidth=3,
        color="black",
    )
    plt.axhline(
        target_temp,
        color="green",
        linewidth=2.5,
        linestyle="--",
        label="目标温度",
    )
    plt.annotate(
        f"目标: {target_temp:.1f}°C",
        xy=(time_x[0], target_temp),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="green"),
    )
    plt.xlabel("时间 (小时)", fontweight="bold")
    plt.ylabel("温度 (°C)", fontweight="bold")
    if y_limits and "temperatures" in y_limits:
        plt.ylim(y_limits["temperatures"])
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
    plt.grid(True, alpha=0.3)
    # plt.title(
    #     f"Building Temperature Profile: {build_type} in {city} ({climate_zone})",
    #     fontweight="bold",
    #     pad=15,
    # )
    plt.tight_layout()
    figs["temperature"] = plt.gcf()

    # Action plot
    if figsize is None:
        figsize = (14, room_count * 0.6)
    plt.figure(figsize=figsize or (14, 6))
    im = plt.imshow(
        actions.T,
        aspect="auto",
        cmap="plasma",
        interpolation="bilinear",
        extent=[time_x[0], time_x[-1], 0, room_count],
        origin="lower",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, orientation="vertical", pad=0.02, label="阀门开度")
    plt.xlabel("时间 (小时)", fontweight="bold")
    plt.ylabel("房间编号", fontweight="bold")
    plt.yticks(range(room_count))
    if y_limits and "actions" in y_limits:
        plt.ylim(y_limits["actions"])
    # plt.title(
    #     f"Building Control Actions: {build_type} in {city} ({climate_zone})",
    #     fontweight="bold",
    #     pad=15,
    # )
    plt.tight_layout()
    figs["action"] = plt.gcf()

    # Performance Score plot
    plt.figure(
        figsize=(7, 4) if figsize is None else (figsize[0] / 2, figsize[1] / 2.5)
    )
    plt.plot(time_x, rewards, linewidth=2.5, color="darkblue")
    plt.xlabel("时间 (小时)", fontweight="bold", fontsize=12)
    plt.ylabel("性能得分", fontweight="bold", fontsize=12)
    if y_limits and "rewards" in y_limits:
        plt.ylim(y_limits["rewards"])
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.axhline(y=0, color="gray", linestyle="-", alpha=0.4)

    # Calculate and plot mean reward
    mean_reward = np.mean(rewards)
    plt.axhline(y=mean_reward, color="red", linestyle="--", label="平均得分")
    plt.annotate(
        f"平均值: {mean_reward:.2f}",
        xy=(time_x[-1], mean_reward),
        xytext=(-10, 0),
        textcoords="offset points",
        fontsize=9,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="lightcoral"),
    )

    # Add min/max annotations
    max_reward = max(rewards)
    min_reward = min(rewards)
    max_idx = np.argmax(rewards)
    min_idx = np.argmin(rewards)
    plt.annotate(
        f"最大值: {max_reward:.2f}",
        xy=(time_x[max_idx], max_reward),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
    )
    plt.annotate(
        f"最小值: {min_reward:.2f}",
        xy=(time_x[min_idx], min_reward),
        xytext=(5, -15),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7),
    )
    plt.tight_layout()
    figs["performance"] = plt.gcf()

    # Save plots if requested
    if save_plots:
        save_custom_plots(figs, output_dir=output_dir, prefix=f"{sim_id}")

    return figs


# Usage examples:

# 1. For environment data (from simulation):
"""
# Call during simulation
sim_id = "my_simulation_run"
plot_data = (env, args, time_x, state_array, action_array, reward_list, Parameter)
figs = plot_building_simulation(
    data=plot_data,
    is_env_data=True,
    sim_id=sim_id,
    save_plots=True
)
"""

# 2. For loaded simulation data:
"""
# Load a simulation and create plots
sim_data = load_simulation("simulation_id")
figs = plot_building_simulation(
    data=sim_data,
    is_env_data=False,
    custom_style='ggplot',
    y_limits={'temperatures': (18, 30), 'actions': (0, 1), 'rewards': (-10, 10)},
    figsize=(16, 12),
    save_plots=True
)
"""

# 3. Update save_results function to use this:
"""
def save_results(env, args, reward_list, reward_breakdown_list, Parameter):
    # ...existing code...
    
    # Replace _generate_plots with our new function
    plot_data = (env, args, time_x, state_array, action_array, reward_list, Parameter)
    plot_building_simulation(
        data=plot_data,
        is_env_data=True,
        sim_id=sim_id,
        save_plots=True
    )
    
    return sim_id
"""
