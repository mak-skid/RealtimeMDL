import os
import pickle
from sedona.spark import *
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType
from pyspark.ml.torch.distributor import TorchDistributor
from mdl.mdl_train import createMDLModelAndTrain
from utils.datapreprocessing_utils import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from lstms.lstm_train import createModelAndTrain
from train.train_distributed import createDistributedModelAndTrain
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

data_path = "NGSIM_Data.csv"

df = sedona.read.csv(data_path, header=True, schema=get_original_schema())
df = us101_filter(df)
df = df.withColumns({"ElapsedTime": F.col("Global_Time") - 1113433135300 - 5413844400})

start = 2229.5 # split the test and train data 80%:20%
#train_df = df.filter(df["ElapsedTime"] < start*1000)
test_df = df.filter(df["ElapsedTime"] >= start*1000) \
            .sort("ElapsedTime") \
            .select(
                "Global_Time",
                "ElapsedTime", 
                "Vehicle_ID", 
                "Global_X", 
                "Global_Y", 
                "Local_X", 
                "Local_Y", 
                "v_Vel", 
                "v_Acc", 
                "Lane_ID", 
                "Location"
            )

#train_df.drop("ElapsedTime").coalesce(1).write.csv("train_data.csv", header=True)
test_df.coalesce(1).write.csv("test_data.csv", header=True)

sedona.stop()