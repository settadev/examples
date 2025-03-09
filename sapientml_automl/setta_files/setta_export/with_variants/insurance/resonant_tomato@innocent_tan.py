import pandas as pd
from sklearn.metrics import r2_score, f1_score

def evaluate(sml, data, target_column, score_fn_info):
    y_true = data[target_column].reset_index(drop=True)
    data.drop([target_column], axis=1, inplace=True)
    y_pred = sml.predict(data)
    y_pred = y_pred[target_column].rename(f"{target_column}_pred")
    pd.concat([y_pred, y_true], axis=1)
    score_fn = score_fn_info.pop("fn")
    print(f"Score: {score_fn(y_true, y_pred, **score_fn_info)}")

def train_and_evaluate():
    version = ($Global Variables$version).split("@")[1]
    $SETTA_GENERATED_PYTHON
    evaluate(sml, test_data, target_column, score_fn_info)


train_and_evaluate()