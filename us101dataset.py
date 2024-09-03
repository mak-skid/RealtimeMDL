import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from utils.datapreprocessing_utils import *


class US101Dataset(Dataset):
    def __init__(
        self,
        df: DataFrame, 
        start: int, 
        end: int, 
        timewindow: int, 
        num_section_splits: int, 
        history_len: int,
        predict_len: int,
        num_skip: int = 0,
        with_ramp: bool = True
    ):
        if (end - start) < timewindow:
            raise ValueError("Time window must be less than the difference between start and end time")
        
        self.with_ramp = with_ramp
        self.num_lanes = 6 if with_ramp else 5
        self.num_sections = num_section_splits + 1
        self.history_len = history_len
        self.predict_len = predict_len
        self.with_ramp = with_ramp
        self.timewindow = timewindow
        self.num_skip = num_skip
        self.start = start
        self.end = end

        timewindow_id = (end - start) // self.timewindow
        num_timewindows = round(timewindow_id)
        print("Number of timewindows: ", num_timewindows + 1)

        self.num_samples = num_timewindows - history_len - num_skip - predict_len
        if not self.num_samples >= 1:
            raise ValueError("num_samples must be greater than 1")

        # For scaling
        scale = df.agg(
            F.min("avg(v_Vel)"), F.max("avg(v_Vel)"), 
            F.min("avg(v_Acc)"), F.max("avg(v_Acc)"),
            F.max("count")
            ).collect()[0]
        self.scale = {
            "min(avg(v_Vel))": scale["min(avg(v_Vel))"],
            "max(avg(v_Vel))": scale["max(avg(v_Vel))"],
            "min(avg(v_Acc))": scale["min(avg(v_Acc))"],
            "max(avg(v_Acc))": scale["max(avg(v_Acc))"],
            "min(count)": 0, # for some reason, the min(count) is not 0
            "max(count)": scale["max(count)"]
        }
        
        print("scale: ", self.scale)
  
        mat3d = df.rdd.map(lambda row: (row["TimeWindow"], row)) \
            .groupByKey() \
            .map(lambda x: rdd_to_np_matrices(x[0], x[1], self.num_lanes, self.num_sections, self.scale, self.with_ramp)) \
            .sortBy(lambda x: x[0]) \
            .values() \
            .collect()
        time_series = np.stack(mat3d)
        print(f"Timewise stacking: {num_timewindows} Apppended. Shape: ", time_series.shape)
    
        history_data = []
        predict_data = []

        # slide history and predict window by one timewindow
        for end_idx in range(history_len + num_skip + predict_len, num_timewindows): # num_skip is the number of timewindows to skip
            predict_frames = time_series[end_idx-predict_len:end_idx][:][:][:]
            history_frames = time_series[end_idx-history_len-num_skip-predict_len:end_idx-num_skip-predict_len][:][:][:]
            history_data.append(history_frames)
            predict_data.append(predict_frames)

            if num_timewindows - end_idx == 1:
                print(f"{end_idx-predict_len-num_skip-history_len+1} History and Predict sample data created") 

        self.history_data = np.stack(history_data)
        self.predict_data = np.stack(predict_data)

        print("History Data Shape: ", self.history_data.shape)
        print("Predict Data Shape: ", self.predict_data.shape)

        self.X_data = torch.tensor(self.history_data)
        self.Y_data = torch.tensor(self.predict_data)


    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, index: int):
        x_data = self.X_data[index]
        y_data = self.Y_data[index]
        sample = {"x_data": x_data, "y_data": y_data}
        return sample
    
    def getShape(self):
        return self.history_data.shape, self.predict_data.shape