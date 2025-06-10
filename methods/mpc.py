import argparse
import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from BEAR.Controller.MPC_Controller import MPCAgent
from BEAR.Env.env_building import BuildingEnvReal
from methods.utils import load_env, reward_func, setup_logger
from methods.utils.save_results import save_results

matplotlib.use("Agg")  # Or 'TkAgg' for interactive plots

logger = setup_logger(__name__, "mpc")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", type=str, default="OfficeSmall")
    parser.add_argument("--climate-zone", type=str, default="Hot_Dry")
    parser.add_argument("--city", type=str, default="Tucson")
    parser.add_argument("--max-timestep", type=int, default=24)
    parser.add_argument(
        "--time-reso",
        type=int,
        default=3600,
        help="Length of 1 timestep in seconds. Default is 3600 (1 hour).",
    )
    parser.add_argument("--horizon", type=int, default=6, help="MPC prediction horizon")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level")
    args = parser.parse_args()
    return args


def generate_args_string(args):
    """
    Generate a string representation of the arguments in an argparse.Namespace object.
    """
    args_dict = vars(args)  # Convert Namespace to a dictionary
    args_str = ", ".join(
        f"{key}={args_dict[key]!r}"
        for key in [
            "build_type",
            "climate_zone",
            "city",
            "max_timestep",
            "time_reso",
            "horizon",
            "noise",
        ]
    )
    return f"Arguments({args_str})"


def init_mpc_controller(env, args):
    """Initialize the MPC controller"""
    mpc_controller = MPCAgent(
        env,
        gamma=env.gamma,
        planning_steps=args.horizon,
        safety_margin=0.96,
        noise=0.0,
    )
    return mpc_controller


def evaluate_MPC(env: BuildingEnvReal, Parameters, args, params):
    """Evaluate the MPC controller"""
    mpc_controller = init_mpc_controller(env, args)
    noised_env, Parameters = load_env(params, reward_func, args.noise)
    obs = noised_env.reset()
    logger.info("Initial observation: %s", obs)

    reward_list = []
    reward_breakdown_list = []
    action_list = []

    pbar = tqdm(range(args.max_timestep), desc="Testing")
    for i in pbar:
        action, s = mpc_controller.predict(env)  # prediction has no noise
        obs, reward, done, truncated, info = noised_env.step(
            action
        )  # real env has noise
        reward_breakdown = info["reward_breakdown"]
        reward_list.append(reward)
        reward_breakdown_list.append(reward_breakdown)
        action_list.append(action)
        pbar.set_postfix({"reward": reward})

    logger.info("reward_mean: {}".format(np.mean(reward_list)))
    logger.info("reward_std: {}".format(np.std(reward_list)))

    # Save results
    id = save_results(noised_env, args, reward_list, reward_breakdown_list, Parameters)
    logger.info(f"Results saved to: {id}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.title(f"MPC Performance ({args.build_type}_{args.climate_zone}_{args.city})")
    plt.plot(reward_list, label="Reward")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    os.makedirs("results/figures", exist_ok=True)
    plt.savefig(
        f"results/figures/MPC_performance_{args.build_type}_{args.climate_zone}_{args.city}.png"
    )
    plt.close()

    return reward_list, reward_breakdown_list, action_list


if __name__ == "__main__":
    args = parse_args()
    args.model = "mpc"
    logger.info("args: %s", vars(args))

    params = {
        "Building": args.build_type,
        "Weather": args.climate_zone,
        "Location": args.city,
        "time_reso": args.time_reso,
    }

    logger.debug("Params: %s", params)

    env, Parameter = load_env(params, reward_func, noise=0.0)
    reward_list, reward_breakdown_list, action_list = evaluate_MPC(
        env, Parameter, args, params
    )
