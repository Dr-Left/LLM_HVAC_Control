from typing import Tuple

import numpy as np

from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator


def load_env(
    params, reward_func, noise=0.0
) -> Tuple[BuildingEnvReal, ParameterGenerator]:
    """
    Load the environment and parameters.
    Reset the environment with a random initial temperature. (between 21 and 23 degrees Celsius)
    Target temperature is set to 22 degrees Celsius.

    Args:
        params: Parameters for the environment.
        reward_func: Function to compute the reward.

    Returns:
        Tuple[BuildingEnvReal, ParameterGenerator]: The environment and parameters.
    """
    Parameter = ParameterGenerator(
        **params, root="BEAR/Data/"
    )  # Description of ParameterGenerator in bldg_utils.py
    # 定义reward分解的键值
    reward_breakdown_keys = ["action_contribution", "error_contribution"]

    # Create environment
    env = BuildingEnvReal(
        Parameter,
        user_reward_function=reward_func,
        reward_breakdown_keys=reward_breakdown_keys,
        noise=noise,
    )
    # Initialize with user-defined indoor temperature
    T_initial = np.random.uniform(21, 23, env.roomnum)
    env.reset(options={"T_initial": T_initial})
    return env, Parameter
