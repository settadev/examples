from datasets import load_dataset
from $callback_fns$import_path import LossCallback

dataset = load_dataset("stanfordnlp/imdb", split="train[:250]")
loss_callback = LossCallback($config$version)

$SETTA_GENERATED_PYTHON

trainer.train()