import numpy as np

from methods.utils import History, chat, setup_logger

from .state_translator import translate_state

from typing import Union

logger = setup_logger(__name__, "history_translator")


class HistoryTranslator:
    def __init__(self, env, args, target_temp):
        self.env = env
        self.args = args
        self.target_temp = target_temp

    def select_history(
        self, history: list[History], method="highest_reward"
    ) -> list[int]:
        """
        Select the history with most and least rewards to enable the model to learn better.
        Return the indices of the selected history.
        """
        if len(history) < 4:
            return list(range(len(history)))
        else:
            indexes = []

        if method == "highest_reward":
            indexed_history = list(enumerate(history))
            indexed_history.sort(
                key=lambda x: x[1].reward,
                reverse=True,
            )
            # indexes += [indexed_history[i][0] for i in range(4)]
            # choose the 4 random history from the top 25%
            N = max(len(indexed_history) // 4, 4)
            indexes += [indexed_history[i][0] for i in range(N)]
            np.random.shuffle(indexes)
            indexes = indexes[:4]
            return indexes
        elif method == "random":
            indexes = list(range(len(history)))
            np.random.shuffle(indexes)
            return indexes[:4]
        else:
            raise ValueError(f"Unknown history method: {method}")

    def translate(self, history: list[History], epoch: int) -> str:
        if len(history) == 0 or self.args.history_method == "none":
            return ""
        selected_indexes = self.select_history(history, method=self.args.history_method)
        assert len(selected_indexes) > 0, "History is empty."
        prompt = "These are some examples giving you an idea how actions affect the state of the building:"
        prompt += "\n<start of examples>\n"
        for i in range(len(selected_indexes)):
            prompt += f"""
**Example {i}**: (At Epoch {selected_indexes[i]})\n
Environment before taking the action:
"""
            prompt += translate_state(
                room_num=self.env.roomnum,
                observation=history[selected_indexes[i]].prior_state,
                target_temp=self.target_temp,
                room_names=self.env.room_names,
            )
            prompt += """
Action taken by the HVAC control: {}
Reward: {:.2f} 
After taking the action, the state changes to:
""".format(
                [int(a) for a in history[selected_indexes[i]].action],
                history[selected_indexes[i]].reward,
            )
            prompt += "\n".join(
                [
                    "{}: {:.2f} degrees Celsius".format(
                        (
                            self.env.room_names[j]
                            if j < len(self.env.room_names)
                            else f"Room {j}"
                        ),
                        history[selected_indexes[i]].post_state[j],
                    )
                    for j in range(self.env.roomnum)
                ]
            )
            prompt += "\n"
        prompt += "<end of examples>\n"
        return prompt


class HistoryTranslatorWithFeedback(HistoryTranslator):
    def __init__(self, env, args, target_temp):
        super().__init__(env, args, target_temp)
        self.hindsights = {}

    def get_hindsight(self, index: int, history: list[History]) -> Union[str, None]:
        if index in self.hindsights:
            return self.hindsights[index]
        prompt = f"""
Now some HVAC controllers are trying to make control actions, please act as critic to give feedback on the action taken by the HVAC control.
The larger the positve action, the more heating is applied.

At Epoch {index}\n
Environment before taking the action:
"""
        prompt += translate_state(
            room_num=self.env.roomnum,
            observation=history[index].prior_state,
            target_temp=self.target_temp,
            room_names=self.env.room_names,
        )
        prompt += """
Action taken by the HVAC control: {}
Reward: {:.2f}
After taking the action, the state changes to:
""".format(
            [int(a) for a in history[index].action],
            history[index].reward,
        )
        prompt += "\n".join(
            [
                "{}: {:.2f} degrees Celsius".format(
                    (
                        self.env.room_names[j]
                        if j < len(self.env.room_names)
                        else f"Room {j}"
                    ),
                    history[index].post_state[j],
                )
                for j in range(self.env.roomnum)
            ]
        )
        prompt += "\n"
        prompt += f"""
Please give feedback and advice on the action taken by the HVAC control, to help the HVAC control to make better actions in the future.
Constraint: The feedback should be less than 50 words.
        """
        logger.debug(f"Highsight prompt: {prompt}")
        response = chat(
            system_prompt="You are a critic, you are given the environment before the action and the action taken by the HVAC control, please give feedback on the action. Give specific feedbacks. If temperature is below the target temperature, the action is not high enough, while if temperature is above the target temperature, the action is too high.",
            user_prompt=prompt,
            model=self.args.model,
        )
        self.hindsights[index] = response
        return response

    def translate(self, history: list[History], epoch) -> str:
        if len(history) == 0:
            return ""
        selected_indexes = self.select_history(history, method=self.args.history_method)
        _hindsights = [
            [history[idx], self.get_hindsight(idx, history)] for idx in selected_indexes
        ]
        logger.info(f"hindsights: {_hindsights}")

        # construct new history
        prompt = """
These are some examples giving you an idea how actions affect the state of the building:
<start of examples>\n"""
        for i in range(len(selected_indexes)):
            prompt += f"""
**Example {i}**: (At Epoch {selected_indexes[i]})\n
Environment before taking the action:
"""
            prompt += translate_state(
                room_num=self.env.roomnum,
                observation=history[selected_indexes[i]].prior_state,
                target_temp=self.target_temp,
                room_names=self.env.room_names,
            )
            prompt += """
Action taken by the HVAC control: {}
Reward: {:.2f}
After taking the action, the state changes to:
""".format(
                [int(a) for a in history[selected_indexes[i]].action],
                history[selected_indexes[i]].reward,
            )
            prompt += "\n".join(
                [
                    "{}: {:.2f} degrees Celsius".format(
                        (
                            self.env.room_names[j]
                            if j < len(self.env.room_names)
                            else f"Room {j}"
                        ),
                        history[selected_indexes[i]].post_state[j],
                    )
                    for j in range(self.env.roomnum)
                ]
            )
            prompt += "\n"
            prompt += f"""
    Feedback: {_hindsights[i][1]}
    """
        prompt += "<end of examples>\n"
        return prompt
