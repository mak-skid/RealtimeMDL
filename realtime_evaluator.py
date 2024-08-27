import json
import pandas as pd
from IPython.display import display
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

act_f = open("act_val.json")
pred_f = open("pred_val_10_2.json")

act_data = json.load(act_f)
pred_data = json.load(pred_f)

pred_df = pd.DataFrame.from_records(pred_data)
act_df = pd.DataFrame.from_records(act_data)

joined_df = act_df.merge(pred_df, on="Global_Time", how="inner")
count = joined_df["actual"].count()

act_flattened = np.array(joined_df["actual"].to_list()).reshape(count, 50)
pred_flattened = np.array(joined_df["prediction"].to_list()).reshape(count, 50)
mse = mean_squared_error(act_flattened, pred_flattened)
mae = mean_absolute_error(act_flattened, pred_flattened)
mape = mean_absolute_percentage_error(act_flattened, pred_flattened)
maape = np.average(np.arctan(np.abs(act_flattened - pred_flattened) / np.abs(act_flattened)))
rmse = np.sqrt(mse)

print(f"MAE: {mae}, MAPE: {mape}, MAAPE:{maape}, RMSE: {rmse}")