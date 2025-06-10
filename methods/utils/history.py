class History:
    def __init__(self, action, info, prior_state, post_state):
        self.action = action
        self.info = info
        self.prior_state = prior_state
        self.post_state = post_state
        self.reward = sum(info["reward_breakdown"].values())
