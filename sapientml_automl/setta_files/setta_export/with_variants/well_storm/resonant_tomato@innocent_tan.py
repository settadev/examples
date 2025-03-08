import pandas as pd
from sklearn.metrics import f1_score

$SETTA_GENERATED_PYTHON

test_data = split_data[1]
y_true = test_data[target_column].reset_index(drop=True)
test_data.drop([target_column], axis=1, inplace=True)
y_pred = sml.predict(test_data)
y_pred = y_pred[target_column].rename(f"{target_column}_pred")
print(f"F1 score: {f1_score(y_true, y_pred)}")