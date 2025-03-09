import pandas as pd
from sklearn.metrics import r2_score, f1_score

def evaluate(sml, data, target_column):
    y_true = data[target_column].reset_index(drop=True)
    data.drop([target_column], axis=1, inplace=True)
    y_pred = sml.predict(data)
    y_pred = y_pred[target_column].rename(f"{target_column}_pred")
    pd.concat([y_pred, y_true], axis=1)
    if sml.task.adaptation_metric == "r2":
        score_fn = r2_score
    else:
        score_fn = f1_score
    print(f"Score: {score_fn(y_true, y_pred)}")

def train():
    version = ($Global Variables$version).split("@")[1]
    $SETTA_GENERATED_PYTHON
    return sml, test_data, target_column

sml, test_data, target_column = train()
evaluate(sml, test_data, target_column)

