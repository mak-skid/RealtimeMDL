import os
import pickle
from sedona.spark import *
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType
from pyspark.ml.torch.distributor import TorchDistributor
from utils.datapreprocessing_utils import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from train import createModelAndTrain
from train_distributed import createDistributedModelAndTrain
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

start = 45 # maybe for start, 45 sec and onwards, bc all sections are filled with data
end = 620.2 # max_elapsed_time=620200 ms = 10.3 minutes
predict_len = 1

preprocessing_param = {
    "timewindow": [0.5], #[1, 3, 5], 
    "num_section_splits": [20], #[40, 60], 
    "history_len": [20], #[90, 120, 150],
    "with_ramp": [False],
    "num_skip": 0
    }

for timewindow in preprocessing_param["timewindow"]:
    for num_section_splits in preprocessing_param["num_section_splits"]:
        for history_len in preprocessing_param["history_len"]:
            for with_ramp in preprocessing_param["with_ramp"]:
                print(f"Model building with parameters = (timewindow: {timewindow}, num_section_splits: {num_section_splits}, history_len: {history_len}, with_ramp: {with_ramp})")
                
                if os.path.exists(f"preprocessed_data/us101_section_agg_{num_section_splits}"):
                    us101_7am_section_agg = sedona.read.csv(
                        f"preprocessed_data/us101_section_agg_{num_section_splits}",
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

                    timewindow_agg_df = timewindow_agg(us101_7am_section_agg, start, end, timewindow)

                    # shape of history_data: (num_samples, history/predict_len, num_features, num_lane, num_section_splits)
                    # initialize the dataset
                    full_dataset = US101Dataset(
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
                    schema = StructType([
                        StructField("Vehicle_ID", IntegerType(), True),
                        StructField("Frame_ID", IntegerType(), True),
                        StructField("Total_Frames", IntegerType(), True),
                        StructField("Global_Time", LongType(), True),
                        StructField("Local_X", DoubleType(), True),
                        StructField("Local_Y", DoubleType(), True),
                        StructField("Global_X", DoubleType(), True),
                        StructField("Global_Y", DoubleType(), True),
                        StructField("v_length", DoubleType(), True),
                        StructField("v_Width", DoubleType(), True),
                        StructField("v_Class", IntegerType(), True),
                        StructField("v_Vel", DoubleType(), True),
                        StructField("v_Acc", DoubleType(), True),
                        StructField("Lane_ID", IntegerType(), True),
                        StructField("O_Zone", IntegerType(), True),
                        StructField("D_Zone", IntegerType(), True),
                        StructField("Int_ID", IntegerType(), True),
                        StructField("Section_ID", IntegerType(), True),
                        StructField("Direction", IntegerType(), True),
                        StructField("Movement", IntegerType(), True),
                        StructField("Preceding", IntegerType(), True),
                        StructField("Following", IntegerType(), True),
                        StructField("Space_Headway", DoubleType(), True),
                        StructField("Time_Headway", DoubleType(), True),
                        StructField("Location", StringType(), True)
                    ])

                    data_path = "NGSIM_Data.csv"

                    df = sedona.read.csv(data_path, header=True, schema=schema)
                    df = convert_coordinate_system(df)
                    df = convert_timestamp(df)
                    df = add_distance_and_time_cols(df)
                    df.show(1)
                    print("Data Preprocessing Done")

                    df = df.withColumns({
                        "day": F.dayofmonth(F.col("datetime")),
                        "month": F.month(F.col("datetime")),
                        "hour": F.hour(F.col("datetime")),
                        "year": F.year(F.col("datetime")),
                        })

                    us101 = us101_filter(df)
                    us101_filtered_hour = hour_filter(us101, "us-101", [7, 8])
                    max_elapsed_time = us101_filtered_hour.tail(1)[0]["ElapsedTime"] # for 7am 620200 ms = 10.3 minutes
                    end = max_elapsed_time // 1000 # convert to seconds
                    max_dist = us101_filtered_hour.select("Distance").sort("Distance").tail(1)[0]["Distance"]

                    us101_section_agg = section_agg(us101_filtered_hour, max_dist, num_section_splits)
                    
                    if not os.path.exists(f"preprocessed_data/us101_section_agg_{num_section_splits}"):
                        us101_section_agg.write.csv(f"preprocessed_data/us101_section_agg_{num_section_splits}")

                    timewindow_agg_df = timewindow_agg(us101_section_agg, start, end, timewindow)

                    # shape of history_data: (num_samples, history/predict_len, num_features, num_lane, num_section_splits)
                    # initialize the dataset
                    full_dataset = US101Dataset(
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

                createModelAndTrain(full_dataset=full_dataset, num_skip=preprocessing_param['num_skip'])

                #TorchDistributor(num_processes=2, local_mode=False, use_gpu=False).run(createDistributedModelAndTrain)
