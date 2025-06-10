# llm.py
"""An LLM pipeline for BEAR"""

import argparse
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from BEAR.Env.env_building import BuildingEnvReal
from methods.llm.translators import parse_output
from methods.llm.translators.prompt_utils import PromptGenerator
from methods.utils import History, chat, load_env, reward_func, setup_logger
from methods.utils.save_results import save_results

matplotlib.use("Agg")  # Or 'TkAgg' for interactive plots

logger = setup_logger(__name__, "llm")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", type=str, default="OfficeSmall")
    parser.add_argument("--climate-zone", type=str, default="Hot_Dry")
    parser.add_argument("--city", type=str, default="Tucson")
    parser.add_argument("--model", type=str, default="deepseek")

    parser.add_argument("--max-timestep", type=int, default=240)
    parser.add_argument(
        "--time-reso",
        type=int,
        default=3600,
        help="Length of 1 timestep in seconds. Default is 3600 (1 hour).",
    )
    parser.add_argument("--prompt-style", "-p", type=str, default="cot_last")
    parser.add_argument(
        "--history-method",
        "-hm",
        type=str,
        choices=["random", "highest_reward", "none"],
        default="highest_reward",
    )
    parser.add_argument("--enable-hindsight", action="store_true")

    # experiment 3: robustness
    parser.add_argument("--noise", type=float, default=0.0)
    """
    available prompt styles are:

    cot_first: Chain of Thought and then gives the result
    cot_last: Chain of Thought but first gives the result

    """
    args = parser.parse_args()
    return args


def llm_step(
    env: BuildingEnvReal,
    args: argparse.Namespace,
    history: List[Tuple[int, Dict[str, Any], np.ndarray]],
    epoch: int,
    prompt_generator: PromptGenerator,
):
    system_prompt, user_prompt = prompt_generator.generate_prompts(history, epoch)
    logger.debug(f"System Prompt:\n{system_prompt}")
    logger.debug(f"User Prompt:\n{user_prompt}")
    while True:
        response = chat(system_prompt, user_prompt, model=args.model)
        logger.debug(f"Response:\n{response}")
        action = parse_output(response, env.roomnum)
        if action is not None:
            break
        logger.warning("Action is None, retrying...")
        time.sleep(1)
    action = np.array(action)
    action = np.clip(action, 0, 10)
    logger.debug(f"Parsed Action: {action}")
    state, reward, done, truncated, info = env.step(
        action / 10
    )  # action itself not normalized
    logger.debug(f"State: {state}")
    return action, info, state


def main():
    args = parse_args()
    # log args
    logger.info(args)
    params = {
        "Building": args.build_type,
        "Weather": args.climate_zone,
        "Location": args.city,
        "time_reso": args.time_reso,
    }
    env, Parameter = load_env(params=params, reward_func=reward_func, noise=args.noise)

    # Initialize prompt generator
    prompt_generator = PromptGenerator(args, env)

    reward_list = []
    reward_breakdown_list = []
    pbar = tqdm(range(args.max_timestep), desc="LLM Loop")  # 240 hours
    history = []

    for i in pbar:
        logger.info(f"LLM Loop epoch: {i}")
        prior_state = env.state
        action, info, post_state = llm_step(env, args, history, i, prompt_generator)
        history.append(History(action, info, prior_state, post_state))
        reward_breakdown = info["reward_breakdown"]
        reward = sum(reward_breakdown.values())
        reward_list.append(reward)
        reward_breakdown_list.append(reward_breakdown)
        pbar.set_postfix({"reward": reward})
        logger.info(f"reward: {reward}")
    logger.info("################LLM LOOP is Done############")
    logger.info(f"reward_mean: {np.mean(reward_list)}")
    logger.info(f"reward_std: {np.std(reward_list)}")
    T = len(reward_list)
    N = env.roomnum
    t = np.array([history[i].post_state[0:N] for i in range(T)])
    delta_t = t - env.target[0]
    logger.info(
        f"Average standard deviation of temperature: {np.sqrt(1 / T * 1 / N * np.sum(delta_t**2)):.2f} C"
    )
    actions = np.array([history[i].action for i in range(T)])
    logger.info(
        f"Average HVAC output action: {np.mean(np.sum(actions / 10, axis=1)):.2f}\n Average HVAC output power: {np.mean(np.sum(actions / 10 * env.maxpower, axis=1)) / 1000:.2f} kW"
    )
    id = save_results(env, args, reward_list, reward_breakdown_list, Parameter)
    logger.info(f"Results saved to: {id}")


if __name__ == "__main__":
    main()
