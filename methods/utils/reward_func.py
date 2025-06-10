import numpy as np
import numpy.linalg as LA


def reward_func(self, state, action, error, state_new, T=22.0):
    """
    Compute the reward based on actions, error, and new state (temperatures).

    Parameters:
    state (np.array): Current state (e.g., temperature values).
    action (np.array): Control actions applied (e.g., heating/cooling efforts).
    error (np.array): The error in maintaining desired conditions.
    state_new (np.array): The resulting state after actions (e.g., new temperatures).
    T (float): The target temperature to maintain.

    Returns:
    float: The computed reward value.
    """
    # Initialize the reward
    reward = 0
    self.alpha = 1

    # 计算各个组成部分
    action_contribution = 1 - np.average(np.abs(action))
    error_contribution = self.alpha * (1 - np.average(LA.norm(error, 2) / T))

    # 更新reward_breakdown
    self._reward_breakdown["action_contribution"] = action_contribution
    self._reward_breakdown["error_contribution"] = error_contribution

    # 总reward
    reward = action_contribution + error_contribution

    return reward
