from nltk.tokenize import word_tokenize

from src.state_info import StateInfo


class ChatBot:
    def __init__(self):
        self.state_info = StateInfo()

    def respond(self, query):

        state_name, district_name = self.process_query(query)
        if state_name and district_name:
            return f"{district_name} is a district in {state_name}."
        elif state_name:
            return f"{state_name} is a state."
        else:
            return "Pardon"

    def process_query(self, query):

        # Tokenize the query
        tokens = word_tokenize(query)

        # Example logic to identify state and district names
        state_name = None
        district_name = None
        for token in tokens:
            if token.lower() in self.state_info.data:
                state_name = token.lower()
            elif state_name and token.lower() in self.state_info.data[state_name]:
                district_name = token.lower()
                break

        return state_name, district_name
