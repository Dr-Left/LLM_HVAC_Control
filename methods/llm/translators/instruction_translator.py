def translate_instruction(out_temp, target_temp, args):

    prompt = """Currently, outside temperature is {} the target temperature.
To optimize HVAC control, adhere to the following guidelines:
1. Actions should be represented as a list, with each integer value {}
2. The length of the actions list should correspond to the number of rooms, arranged in the same order.
3. Your goal is to maintain the temperature of each room as close to the target temperature as possible, while minimizing the sum of the actions.
4. Because there is heat transfer between the building and the outside environment, the building temperature will be affected by the outside temperature.(If outside is cooler, the building will be cooler afterwards, and vice versa.)
5. The timestep is {} seconds.
"""
    if out_temp > target_temp:
        prompt = prompt.format(
            "higher than",
            "ranging from -10 to 0. (the larger the absolute value, the more cooling)",
            args.time_reso,
        )
    elif out_temp < target_temp:
        prompt = prompt.format(
            "lower than",
            "ranging from 0 to 10. (the larger the absolute value, the more heating)",
            args.time_reso,
        )
    else:
        prompt = prompt.format(
            "equal to",
            "ranging from -10 to 10 (the larger the absolute value, the more heating or cooling)",
            args.time_reso,
        )
    return prompt
