from datasets import load_dataset
from $scripts["callback_fns"]$import_path import LossCallback

dataset = load_dataset("stanfordnlp/imdb", split="train[:250]")
loss_callback = LossCallback($training["config"]$version)

$SETTA_GENERATED_PYTHON

training["trainer"].train()