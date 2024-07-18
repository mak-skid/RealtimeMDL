import numpy as np
import torch
from torch.utils.data import Dataset
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from datapreprocessing import create_np_matrices


class US101Dataset(Dataset):
    def __init__(
        self,
        df: DataFrame, 
        start: int, 
        end: int, 
        timewindow: int, 
        num_section: int, 
        history_len: int,
        predict_len: int
    ):
        if (end - start) < timewindow:
            raise ValueError("Time window must be less than the difference between start and end time")
        
        num_lanes = 6
        timewindow_id = (end - start) // timewindow
        num_timewindows = round(timewindow_id)
        print("Number of timewindows: ", num_timewindows)

        self.history_len = history_len
        self.predict_len = predict_len
        self.num_samples = num_timewindows - history_len - predict_len
        if not self.num_samples >= 1:
            raise ValueError("num_samples must be greater than 1")

        '''
        vel_df = df.select("TimeWindow", "Section_ID", "Lane_ID", "avg(v_Vel)")
        dens_df = df.select("TimeWindow", "Section_ID", "Lane_ID", "count")
        acc_df = df.select("TimeWindow", "Section_ID", "Lane_ID", "avg(v_Acc)")
        '''

        vel_stack = np.empty(shape=(num_lanes, num_section+1, num_timewindows))
        dens_stack = np.empty(shape=(num_lanes, num_section+1, num_timewindows))
        acc_stack = np.empty(shape=(num_lanes, num_section+1, num_timewindows))

        # 4D array, x: timewindow, y: lane, z: section, w: velocity, density, acceleration features
        # full_data = np.empty(shape=(num_timewindows, num_lanes, num_section+1, 3))

        time_series = []
        history_data = []
        predict_data = []

        for i in range(num_timewindows):
            timewindow_df = df.filter((F.col("TimeWindow") == i)) # use mapPartition for each timewindow
            vel_stack[:, :, i], dens_stack[:, :, i], acc_stack[:, :, i] = create_np_matrices(timewindow_df, num_section)
            time_series.append([vel_stack[:, :, i], dens_stack[:, :, i], acc_stack[:, :, i]])
            print(f"Timewise stacking: {i} TimeWindows appended")

        for end_idx in range(history_len + predict_len, num_timewindows):
            predict_frames = time_series[end_idx-predict_len:end_idx]
            history_frames = time_series[end_idx-predict_len-history_len:end_idx-predict_len]
            history_data.append(history_frames)
            predict_data.append(predict_frames)
            print(f"{end_idx-predict_len-history_len+1} History and Predict sample data created")

        history_data = np.stack(history_data)
        predict_data = np.stack(predict_data)

        self.X_data = torch.tensor(history_data)
        self.Y_data = torch.tensor(predict_data)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, index: int):
        x_data = self.X_data[index]
        y_data = self.Y_data[index]
        sample = {"x_data": x_data, "y_data": y_data}
        return sample