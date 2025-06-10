import argparse
from typing import Any, Dict, List, Tuple

import numpy as np

from BEAR.Env.env_building import BuildingEnvReal
from methods.llm.translators import HistoryTranslator, HistoryTranslatorWithFeedback

from .instruction_translator import translate_instruction
from .meta_translator import translate_meta
from .state_translator import translate_state


class PromptGenerator:
    def __init__(self, args: argparse.Namespace, env: BuildingEnvReal):
        self.args = args
        self.env = env
        self.target_temp = env.target[0]

        if args.enable_hindsight:
            self.history_translator = HistoryTranslatorWithFeedback(
                env, args, self.target_temp
            )
        else:
            self.history_translator = HistoryTranslator(env, args, self.target_temp)

    def generate_prompts(
        self, history: List[Tuple[int, Dict[str, Any], np.ndarray]], epoch: int
    ):
        """Generate system and user prompts for the LLM."""
        system_prompt = translate_meta(self.args, self.env)

        state_info = translate_state(
            room_num=self.env.roomnum,
            observation=self.env.state,
            target_temp=self.target_temp,
            room_names=self.env.room_names,
        )

        instruction_info = translate_instruction(
            out_temp=self.env.OutTemp[self.env.epochs],
            target_temp=self.target_temp,
            args=self.args,
        )

        history_info = self.history_translator.translate(history, epoch)

        if self.args.prompt_style == "cot_first":
            user_prompt = f"""
{instruction_info}

Current Epoch: {epoch}
{history_info}

Based on the feedback of the past actions and the current state of the building, please decide the opening of the valve in each room, to make the temperature closer to the target temperature, and the actions are as close to 0 as possible.
Description: {state_info}
Actions: (Output Example: Reason... Action:[action_1, action_2, action_3, action_4, ...])

Let's think step by step: (Give your final decision in 100 words)
"""
        elif self.args.prompt_style == "cot_last":
            user_prompt = f"""
{instruction_info}

Current Epoch: {epoch}
{history_info}

Based on the feedback of the past actions and the current state of the building, please decide the opening of the valve in each room, to make the temperature closer to the target temperature, and the actions are as close to 0 as possible.
(Reminder: **First** give the list of numbers and **then** give the reason.)
Description: {state_info}
Actions: (Output Example: [1, 0, 1, 0, ...] + Reason)
"""

        return system_prompt, user_prompt
