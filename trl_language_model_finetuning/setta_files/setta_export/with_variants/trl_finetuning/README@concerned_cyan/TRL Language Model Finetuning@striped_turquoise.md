## TRL Language Model Finetuning

This is a basic example of how to configure LLM training runs. 

## How to run

Click the Run button in the top nav bar.


## What each section does

- `trainer`: This is a [`trl.SFTTrainer`](https://huggingface.co/docs/trl/en/sft_trainer) object. It takes care of loading and training the model.
- `config`: This contains all the details of training. In this example, there are 3 versions of this section, which you can view in the versions pane.
- `training_script`: Creates the training dataset, loss-logging callback, config object, and trainer (the config and trainer are created in `$SETTA_GENERATED_PYTHON`), and starts training.
- `callback_fns`: Defines the callback for logging losses to Setta.
- `Global Param Sweep`: Defines a sweep over config versions, so that all three versions run at the same time in parallel.