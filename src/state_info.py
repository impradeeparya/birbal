import json
import os


class StateInfo:
    def __init__(self):
        json_file_path = "data/states-and-districts.json"

        print("Absolute path:", os.path.abspath(json_file_path))

        with open(json_file_path, 'r') as states_and_districts:
            states = json.load(states_and_districts)
            self.data = states['states']
        print(len(self.data), " states data loaded")
