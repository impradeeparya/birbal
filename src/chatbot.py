from src.state_info_trainer import StateTrainer


class ChatBot:
    def __init__(self):
        self.state_trainer = StateTrainer()

    def respond(self, query):
        state_name, district_name = self.process_query(query)
        if state_name and district_name:
            return f"{district_name} is a district in {state_name}."
        elif state_name:
            return f"{state_name} is a state."
        else:
            return "Pardon"

    def process_query(self, query):
        # Example logic to identify state and district names
        state_name, districts = self.state_trainer.fetch_token(query)
        print(state_name, districts)
        return state_name, None
