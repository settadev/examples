from transformers import TrainerCallback
from setta.cli import Setta, SettaList

class LossCallback(TrainerCallback):
    def __init__(self, name):
        self.name = name
        self.logger = Setta()
        self.losses = {"loss": []}

    def on_log(self, args, state, control, logs, **kwargs):
        if 'loss' in logs:
            self.losses["loss"].append(logs['loss'])
            self.logger.log({self.name: SettaList(self.losses)})