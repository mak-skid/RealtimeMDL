import os
import pickle
from sedona.spark import *
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType
from pyspark.ml.torch.distributor import TorchDistributor
from mdl.mdl_train import createMDLModelAndTrain
from utils.datapreprocessing_utils import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from us101dataset import US101Dataset


config = SedonaContext.builder() \
    .master("local[*]") \
    .appName("SedonaSample") \
    .config('spark.jars.packages', 
            'org.apache.sedona:sedona-spark-3.5_2.12:1.6.0,'
            'org.datasyslab:geotools-wrapper:1.6.0-28.2'
        ) \
    .getOrCreate()
sedona = SedonaContext.create(config)
print("Sedona Initialized")

realtime_mode = False
start = 45 # maybe for start, 45 sec and onwards, bc all sections are filled with data
end = 2229.5 if realtime_mode else 2772.5 # after splitting the test and train data 80%:20%
predict_len = 1

preprocessing_param = {
    "timewindow": [10], #[0.5, 1, 3, 5], 
    "num_section_splits": [9], #[40, 60], 
    "history_len": [6], #[20, 90, 120, 150],
    "with_ramp": [False],
    "num_skip": 1 # 10
    }


def check_max_elapsed_time(df):
    max_elapsed_time = df.select("ElapsedTime").tail(1)[0]["ElapsedTime"]
    print(f"Max Elapsed Time: {max_elapsed_time}")
    return max_elapsed_time

for timewindow in preprocessing_param["timewindow"]:
    for num_section_splits in preprocessing_param["num_section_splits"]:
        for history_len in preprocessing_param["history_len"]:
            for with_ramp in preprocessing_param["with_ramp"]:
                print(f"Model building with parameters = (timewindow: {timewindow}, num_section_splits: {num_section_splits}, history_len: {history_len}, with_ramp: {with_ramp})")
                
                if os.path.exists(f"dataset/preprocessed_data/us101_section_agg_{num_section_splits}"):
                    us101_section_agg = sedona.read.csv(
                        f"dataset/preprocessed_data/us101_section_agg_{num_section_splits}",
                        schema = StructType([
                            StructField("ElapsedTime", LongType(), False), 
                            StructField("Section_ID", IntegerType(), False),
                            StructField("Lane_ID", IntegerType(), False), 
                            StructField("avg(v_Vel)", DoubleType(), False), 
                            StructField("avg(v_Acc)", DoubleType(), False), 
                            StructField("count", IntegerType(), False)
                        ])
                    )
                    print('Preprocessed data loaded')

                    timewindow_agg_df = timewindow_agg(us101_section_agg, start, end, timewindow)

                    # shape of history_data: (num_samples, history/predict_len, num_features, num_lane, num_section_splits)
                    # initialize the dataset
                    train_dataset = US101Dataset(
                        timewindow_agg_df, 
                        start, 
                        end, 
                        timewindow, 
                        num_section_splits, 
                        history_len, 
                        predict_len, 
                        num_skip=preprocessing_param['num_skip'],
                        with_ramp=with_ramp
                        )    
                else:
                    data_path = "train_data.csv" if realtime_mode else "NGSIM_Data.csv"

                    df = sedona.read.csv(f"dataset/{data_path}", header=True, schema=get_original_schema())
                    df = us101_filter(df)
                    df = convert_to_mph(df) if data_path == "NGSIM_Data.csv" else df
                    df = convert_coordinate_system(df)
                    df = convert_timestamp(df)
                    df = add_distance_and_time_cols(df)

                    print("Data Preprocessing Done")

                    df = df.withColumns({
                        "day": F.dayofmonth(F.col("datetime")),
                        "month": F.month(F.col("datetime")),
                        "hour": F.hour(F.col("datetime")),
                        "year": F.year(F.col("datetime")),
                        })

                    us101_filtered_hour = hour_filter(df, "us-101", [7, 8])
                    max_elapsed_time = check_max_elapsed_time(us101_filtered_hour) # for 7am 620200 ms = 10.3 minutes
                    end = max_elapsed_time // 1000 # convert to seconds
                    max_dist = us101_filtered_hour.select("Distance").sort("Distance").tail(1)[0]["Distance"] # 2224.6633212502065 

                    us101_section_agg = section_agg(us101_filtered_hour, max_dist, num_section_splits)
                    
                    if not os.path.exists(f"dataset/preprocessed_data/us101_section_agg_{num_section_splits}"):
                        us101_section_agg.write.csv(f"dataset/preprocessed_data/us101_section_agg_{num_section_splits}")

                    timewindow_agg_df = timewindow_agg(us101_section_agg, start, end, timewindow)

                    # shape of history_data: (num_samples, history/predict_len, num_features, num_lane, num_section_splits)
                    # initialize the dataset
                    train_dataset = US101Dataset(
                        timewindow_agg_df, 
                        start, 
                        end, 
                        timewindow, 
                        num_section_splits, 
                        history_len, 
                        predict_len, 
                        num_skip=preprocessing_param['num_skip'],
                        with_ramp=with_ramp
                        )
                    
                createMDLModelAndTrain(train_dataset=train_dataset, num_features=1, num_skip=preprocessing_param['num_skip'], realtime_mode=realtime_mode)