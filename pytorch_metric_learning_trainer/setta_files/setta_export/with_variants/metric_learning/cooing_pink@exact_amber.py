$SETTA_GENERATED_PYTHON

models["trunk"].fc = torch.nn.Identity()

trainer.train(num_epochs=5)