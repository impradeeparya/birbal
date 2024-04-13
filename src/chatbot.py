from src.state_info import StateInfo


class ChatBot:
    def __init__(self):
        self.state_info = StateInfo()

    def respond(self, user_input):
        state_info = {}
        for state in self.state_info.data:
            if state['state'] == user_input:
                state_info = state
        return state_info
