import argparse
import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.ppo import MlpPolicy
from tqdm import tqdm

from BEAR.Env.env_building import BuildingEnvReal
from methods.utils import load_env, reward_func, setup_logger
from methods.utils.save_results import save_results  # Import the save_results function

matplotlib.use("Agg")  # Or 'TkAgg' for interactive plots

logger = setup_logger(__name__, "ppo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", type=str, default="OfficeSmall")
    parser.add_argument("--climate-zone", type=str, default="Hot_Dry")
    parser.add_argument("--city", type=str, default="Tucson")

    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--max-timestep", type=int, default=24)
    parser.add_argument(
        "--time-reso",
        type=int,
        default=3600,
        help="Length of 1 timestep in seconds. Default is 3600 (1 hour).",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--continuing", action="store_true")
    parser.add_argument("--noise", type=float, default=0.0)
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
            "train_steps",
            "max_timestep",
            "time_reso",
        ]
    )
    return f"Arguments({args_str})"


def init_model(env, args):
    model = PPO(
        MlpPolicy,
        env,
        n_epochs=10,
        gamma=0.99,
        learning_rate=args.learning_rate,
        clip_range=0.2,
        verbose=0,
        device="cpu",
    )

    return model


def train_PPO(env, Parameter, args):
    model_path = f"models/PPO_{args.build_type}_{args.climate_zone}_{args.city}.zip"
    rewardlist_path = (
        f"rewards/PPO_{args.build_type}_{args.climate_zone}_{args.city}.csv"
    )

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("rewards"):
        os.makedirs("rewards")

    if os.path.exists(model_path) and args.continuing:
        model = PPO.load(model_path)
        rewardlist = pd.read_csv(rewardlist_path)
        rewardlist = rewardlist.iloc[:, 1].values.tolist()
        logger.info("Continuing training from previous model.")
    else:
        logger.info("Initializing a new model.")
        model = init_model(env, args)
        rewardlist = []

    env.reset()
    model.set_env(env)

    action_record = []

    pbar = tqdm(range(args.train_steps), desc="Training")
    for i in pbar:
        model.learn(total_timesteps=1000)
        rw = 0
        vec_env = model.get_env()
        obs = vec_env.reset()
        for j in range(args.max_timestep):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            rw += rewards
        pbar.set_postfix({"reward": rw.item() / args.max_timestep})
        rewardlist.append(rw / args.max_timestep)
        action_record.append(np.array(env.actionlist).sum(axis=1))

    logger.info("################TRAINING is Done############")
    model.save(model_path)
    os.makedirs(os.path.dirname(rewardlist_path), exist_ok=True)
    pd.DataFrame(rewardlist).to_csv(rewardlist_path)
    return model, rewardlist, action_record


def evaluate_PPO(env: BuildingEnvReal, Parameters, args):
    model = PPO.load(f"models/PPO_{args.build_type}_{args.climate_zone}_{args.city}")
    # env.reset(epochs=args.max_timestep * args.time_reso * 3)
    env.reset()
    model.set_env(env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    logger.info("Initial observation: %s", obs)

    reward_list = []
    reward_breakdown_list = []

    pbar = tqdm(range(args.max_timestep), desc="Testing")
    for i in pbar:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        reward_breakdown = info[0]["reward_breakdown"]
        reward = rewards.item()
        reward_list.append(reward)
        reward_breakdown_list.append(reward_breakdown)
        pbar.set_postfix({"reward": reward})

    logger.info("reward_mean: {}".format(np.mean(reward_list)))
    logger.info("reward_std: {}".format(np.std(reward_list)))

    # Save results using the same function as in llm.py
    id = save_results(env, args, reward_list, reward_breakdown_list, Parameters)
    logger.info(f"Results saved to: {id}")

    return reward_list, reward_breakdown_list


if __name__ == "__main__":
    set_random_seed(seed=1234)
    args = parse_args()
    args.model = "ppo"
    logger.info("args: %s", vars(args))

    params = {
        "Building": args.build_type,
        "Weather": args.climate_zone,
        "Location": args.city,
        "time_reso": args.time_reso,
    }

    logger.debug("Params: %s", params)
    logger.debug("train_steps: %d", args.train_steps)

    assert args.eval or args.noise == 0.0, "Noise must be 0.0 when training"

    env, Parameter = load_env(params, reward_func, noise=args.noise)

    if not args.eval:
        model, rewardlist, action_record = train_PPO(env, Parameter, args)

        # Plot training rewards
        plt.figure(figsize=(10, 6))
        plt.title(
            f"PPO Training Reward ({args.build_type}_{args.climate_zone}_{args.city})"
        )
        plt.plot(rewardlist)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

        os.makedirs("results/figures", exist_ok=True)
        plt.savefig(
            f"results/figures/PPO_training_reward_{args.build_type}_{args.climate_zone}_{args.city}.png"
        )
        plt.close()

    # Evaluate the model
    evaluate_PPO(env, Parameter, args)
