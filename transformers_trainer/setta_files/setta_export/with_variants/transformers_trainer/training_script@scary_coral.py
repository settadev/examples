def get_tokenize_function(tokenizer, **kwargs):
    def tokenize_function(examples):
        return tokenizer(examples["text"], **kwargs)
    return tokenize_function

$SETTA_GENERATED_PYTHON
