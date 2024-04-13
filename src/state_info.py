import json
import os


class StateInfo:
    def __init__(self):
        json_file_path = "data/states-and-districts.json"

        print("Absolute path:", os.path.abspath(json_file_path))

        with open(json_file_path, 'r') as states_and_districts:
            self.states = json.load(states_and_districts)
            self.data = self.populate_states_data()
        print(len(self.data), " states data loaded")

    def populate_states_data(self):
        data = {}
        for state in self.states['states']:
            data[state['state'].lower()] = [s.lower() for s in state['districts']]
        return data
