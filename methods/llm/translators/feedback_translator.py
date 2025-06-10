def translate_feedback(reward, actions, room_num, states_after, target_temp):
    """
    Deprecated
    """

    prompt = """Reward: {}
Actions: {}
Comments: After taking the above actions, temperature in each room becomes:
{}
{}"""
    reward = int(reward * 10)
    actions = [int(a * 10) for a in actions]
    temps = "\n".join(
        [
            "Room {}: {} degrees Celsius".format(i, states_after[i])
            for i in range(room_num)
        ]
    )
    out_temp = states_after[room_num]
    comment = "\n".join(
        [
            "The action for Room {} shall be {} as its temperature is {} than the target temperature.".format(
                i,
                "decreased" if states_after[i] > target_temp else "increased",
                "higher" if states_after[i] > target_temp else "lower",
            )
            for i in range(room_num)
        ]
    )
    prompt = prompt.format(reward, actions, temps, comment)
    return prompt
