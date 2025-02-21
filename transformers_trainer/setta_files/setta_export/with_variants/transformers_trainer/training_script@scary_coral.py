import numpy as np
from callbacks import LossCallback

def get_tokenize_function(tokenizer, **kwargs):
    def tokenize_function(examples):
        return tokenizer(examples["text"], **kwargs)
    return tokenize_function

def get_compute_metrics(metric):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    return compute_metrics

training_args_version = $training_setup["training_args"]$version
dataset_version = $dataset_setup["dataset"]$version
callback_name = f"{training_args_version}_{dataset_version}"
loss_callback = LossCallback(callback_name)

$SETTA_GENERATED_PYTHON

training_setup["trainer"].train()
