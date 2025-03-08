import pandas as pd
from sklearn.metrics import r2_score

$SETTA_GENERATED_PYTHON

y_true = test_data[target_column].reset_index(drop=True)
test_data.drop([target_column], axis=1, inplace=True)
y_pred = sml.predict(test_data)
y_pred = y_pred[target_column].rename(f"{target_column}_pred")
pd.concat([y_pred, y_true], axis=1)
print(f"R2 score: {r2_score(y_true, y_pred)}")