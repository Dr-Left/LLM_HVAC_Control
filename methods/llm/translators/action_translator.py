def translate_action(actions):
    prompt = "Actions: {}".format([int(action) for action in actions])
    return prompt
