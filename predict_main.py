import os
import pickle
from sedona.spark import *
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType
from pyspark.ml.torch.distributor import TorchDistributor
from utils.datapreprocessing_utils import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

from us101dataset import US101Dataset


class SparkJob:
    def section_agg(self, df: DataFrame, max_dist: int, num_section_splits: int) -> DataFrame:
        """
        df: pyspark dataframe containing the US101 dataset

        Returns
        --------
        df: pyspark dataframe containing the aggregated values of avg(v_Vel), avg(v_Acc), and count, 
            grouped by timestmap, Section_ID, and Lane_ID
        max_dist: int, maximum distance in the dataset
        num_section_splits: int, number of splits to perform on road section of the dataset
        """
        df = df \
            .withColumn("Section_ID", 
                F.round(
                    (F.col("Distance") / F.lit(max_dist // num_section_splits)).cast("integer")
                ) # gives a Section ID to each datapoint 
            ) \
            .select("timestamp", "Lane_ID", "v_Vel", "Section_ID") \
            .groupBy("timestamp", "Section_ID", "Lane_ID") \
            .agg(
                F.round(F.avg("v_Vel"), 1).alias("avg(v_Vel)")
            )
        return df
    
    def init_job(self):
        config = SedonaContext.builder() \
            .master("local[*]") \
            .appName("SedonaSample") \
            .config('spark.jars.packages', 
                    'org.apache.sedona:sedona-spark-3.5_2.12:1.6.0,'
                    'org.datasyslab:geotools-wrapper:1.6.0-28.2,'
                    'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,'
                    'org.apache.kafka:kafka-clients:3.8.0,'
                    'org.apache.spark:spark-token-provider-kafka-0-10_2.12:3.5.0,'
                    ) \
            .config('spark.sql.streaming.statefulOperator.checkCorrectness.enabled', 'false') \
            .getOrCreate()
        sedona = SedonaContext.create(config)
        print("Sedona Initialized")

        start = 2229.5 # after spliting the test and train data 80%:20%
        end = 2772 # max_elapsed_time=2772000 ms = 46.2 minutes if hour is 7-8am
        predict_len = 1

        timewindow = 0.5
        num_section_splits = 20
        history_len = 20
        with_ramp = False
        num_skip = 10

        df = sedona.readStream \
            .format('kafka') \
            .option('kafka.bootstrap.servers', 'localhost:9092') \
            .option('subscribe', 'us101') \
            .option('startingOffsets', 'earliest') \
            .load() 

        parsed_df = df \
            .select(
                F.col("timestamp"),
                F.from_json(F.col("value").cast("string"), schema=get_test_schema()).alias("parsed_value")
            ) \
            .select('*', F.inline("parsed_value")).drop("parsed_value", "v_Acc") 
        
        dist_added_df = parsed_df \
            .withColumns({"Distance": F.sqrt(F.pow(F.col("Local_X"), 2) + F.pow(F.col("Local_Y"), 2))}) \
            .drop("Local_X", "Local_Y")   
        max_dist = 2224.6633212502065 
        section_agg_df = dist_added_df \
            .withColumn("Section_ID", 
                F.round(
                    (F.col("Distance") / F.lit(max_dist // num_section_splits)).cast("integer")
                ) # gives a Section ID to each datapoint 
            ) \
            .select("timestamp", "Lane_ID", "v_Vel", "Section_ID") \
            .groupBy("timestamp", "Section_ID", "Lane_ID") \
            .agg(
                F.round(F.avg("v_Vel"), 1).alias("avg(v_Vel)")
            )
        
        section_agg_df = self.section_agg(dist_added_df, max_dist, num_section_splits)
        
        timewindow_agg_df = section_agg_df \
            .withWatermark("timestamp", f"{timewindow} second") \
            .groupBy(F.window(F.col("timestamp"), f"{timewindow} second"), "Section_ID", "Lane_ID") \
            .agg(
                F.round(F.avg("avg(v_Vel)"), 1).alias("avg(v_Vel)")
            ) \
        
        #query = timewindow_agg_df \
        query = timewindow_agg_df \
            .writeStream \
            .outputMode("update") \
            .format("console") \
            .option("truncate", "false") \
            .start()

        query.awaitTermination()

        """
        train_dataset = US101Dataset(
            timewindow_agg_df, 
            start, 
            end, 
            timewindow, 
            num_section_splits, 
            history_len, 
            predict_len, 
            num_skip=num_skip,
            with_ramp=with_ramp
            )

        query = parsed_df.writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", "false") \
            .start()
        query.awaitTermination()
        """

SparkJob().init_job()