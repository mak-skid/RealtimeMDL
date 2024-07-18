import os
import pickle
from sedona.spark import *
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType
from datapreprocessing import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from train import createModelAndTrain
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

sc = sedona.sparkContext

if os.path.exists("us101preprocessed.pkl"):
    with open('us101preprocessed.pkl', 'rb') as file:
        full_dataset = pickle.load(file)
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

    def filterUS101_7am(df: DataFrame):
        us101 = df \
            .filter(F.col("Location") == "us-101") \
            .withColumn("Lane_ID", 
                        F.when(
                            F.col("Lane_ID").isin([7, 8]), 6) # merge lanes 7 and 8 into 6
                            .otherwise(F.col("Lane_ID"))
                            )
        us101_7am = us101.filter((F.col("hour") == 7)).sort("datetime") \
            .select(
                "Location", 
                "ElapsedTime", 
                "hour", 
                "Distance", 
                "Vehicle_ID", 
                "Lane_ID", 
                "v_Vel", 
                "v_Acc",  
                "lat", 
                "lon") \
            .withColumn("ElapsedTime", F.col("ElapsedTime") - 5413844400) \
            .sort("ElapsedTime")
        
        us101_7am.show(1)
        print("US101 7am Data Filtered")
        return us101_7am

    us101_7am = filterUS101_7am(df)    
    max_elapsed_time = us101_7am.tail(1)[0]["ElapsedTime"] # 620200 ms = 10.3 minutes
    max_dist = us101_7am.select("Distance").sort("Distance").tail(1)[0]["Distance"]

    start, end, timewindow = 0, max_elapsed_time//1000, 5
    num_section = 40
    history_len = 60
    predict_len = 10

    """
    start, end, timewindow = 0, 300, 30
    num_section = 20
    history_len = 5
    predict_len = 1
    """

    us101_7am_section_agg = section_agg(us101_7am, max_dist, num_section)
    timewindow_agg_df = timewindow_agg(us101_7am_section_agg, start, end, timewindow=5)

    # shape of history_data: (num_samples, history/predict_len, num_features, num_lane, num_section)
    full_dataset = US101Dataset(timewindow_agg_df, start, end, timewindow, num_section, history_len, predict_len)

    with open('us101preprocessed.pkl', 'wb') as file:
        pickle.dump(full_dataset, file)

createModelAndTrain(
    full_dataset=full_dataset
)
