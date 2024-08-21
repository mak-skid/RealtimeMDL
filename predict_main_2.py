import os
import pickle
import keras
from sedona.spark import *
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType, TimestampType, DecimalType, ArrayType, FloatType
from pyspark.ml.torch.distributor import TorchDistributor
from mdl.mdl_model import get_MDL_model
from utils.datapreprocessing_utils import *
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import DataFrame

from us101dataset import US101Dataset

class SecondWatermarkAggregator:
    def __init__(self):
        self.bootstrap_servers = 'localhost:9092'    

        self.start = 2229.5 # after spliting the test and train data 80%:20%
        self.end = 2772 # max_elapsed_time=2772000 ms = 46.2 minutes if hour is 7-8am
        self.predict_len = 1

        self.timewindow = 0.5
        self.num_section_splits = 9
        self.num_lanes = 5
        self.history_len = 20
        self.num_skip = 10
        self.max_dist = 2224.6633212502065
        self.num_sections = self.num_section_splits + 1

        with_ramp = False
        self.with_ramp_sign = "w" if with_ramp else "wo"
        self.num_features = 1

        self.model = get_MDL_model(self.history_len, self.num_lanes, self.num_sections, self.num_features)
    
    def parse_df(self, df: DataFrame) -> DataFrame:
        key_schema = StructType([
            StructField("timewindow", 
                StructType([
                    StructField("start", TimestampType()),
                    StructField("end", TimestampType())
                ])),
            StructField("start_timestamp", LongType())
        ])
        value_schema = ArrayType(ArrayType(ArrayType(IntegerType(), False), False), False)

        return df \
            .select(
                F.from_json(F.col("key").cast("string"), schema=key_schema).alias("parsed_key"),
                F.from_json(F.col("value").cast("string"), schema=value_schema).alias("3D_mat")
            ) \
            .withColumn("timewindow", F.col("parsed_key.timewindow")) \
            .withColumn("Global_Time", F.col("parsed_key.start_timestamp")) \
            .drop("parsed_key")

    def to_4d_np(self, df: DataFrame) -> DataFrame:  
        return convert_timestamp(df) \
            .withWatermark("datetime", f"{self.timewindow*self.history_len} second") \
            .groupBy(
                F.window(
                    F.col("datetime"), 
                    f"{self.timewindow*self.history_len} second",
                    f"{self.timewindow} second"
                ).alias("timewindow")
            ) \
            .agg(
                F.sort_array(F.collect_list(F.struct("Global_Time", "3D_mat"))).alias("4D_mat")
            ) \
            .withColumn("sorted_4D_mat", F.col("4D_mat.3D_mat")) \
            .select("timewindow", "4D_mat.Global_Time", "sorted_4D_mat") # (history_len, num_lanes, num_sections, num_features)
    
    def real_time_prediction(self, df: DataFrame) -> DataFrame:
        def prediction(rows):
            model_name = f'mdl_model_{self.with_ramp_sign}_{self.timewindow}_{self.num_sections}_{self.history_len}_{self.num_features}_{self.num_skip}'
            #model = keras.saving.load_model(f"mdl/models/{model_name}.keras")
            self.model.load_weights(f"mdl/models/{model_name}.weights.h5")

            if len(rows) < 20: # if there is not enough data to predict
                return np.zeros((self.num_lanes, self.num_sections)).tolist()
            input = rows[:20]
            input_mat = np.array(input).reshape(1, self.history_len, self.num_lanes, self.num_sections, self.num_features)

            if input_mat.shape != (1, self.history_len, self.num_lanes, self.num_sections, self.num_features):
                return np.zeros((self.num_lanes, self.num_sections)).tolist()
            
            pred = self.model.predict(input_mat)
            return pred.reshape(self.num_lanes, self.num_sections).tolist()

        pred_udf = F.udf(prediction, ArrayType(ArrayType(IntegerType())))

        return df \
            .withColumn("prediction", pred_udf(F.col("sorted_4D_mat"))) \
            .select(F.col("timewindow").alias("based_timewindow"), F.col("prediction")) \
        
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

        df = sedona.readStream \
            .format('kafka') \
            .option('kafka.bootstrap.servers', self.bootstrap_servers) \
            .option('subscribe', 'us101_agg1') \
            .option('startingOffsets', 'earliest') \
            .load()
        
        df.printSchema()
        parsed_df = self.parse_df(df)
        test_df = self.to_4d_np(parsed_df)
        pred_df = self.real_time_prediction(test_df)

        query = pred_df \
            .writeStream \
            .outputMode("append") \
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
        """

SecondWatermarkAggregator().init_job()