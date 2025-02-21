import torch

$SETTA_GENERATED_PYTHON

device = torch.device("cuda")
models["models"]["trunk"].fc = torch.nn.Identity()
models["models"] = {k:v.to(device) for k,v in models["models"].items()}

trainer.train(num_epochs=epochs)