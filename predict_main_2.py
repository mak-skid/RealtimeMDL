from sedona.spark import *
from pyspark.sql.types import StructType, StructField, IntegerType, TimestampType, ArrayType
from realtime_predictor import RealTimePredictor
from utils.datapreprocessing_utils import *
from pyspark.sql import functions as F
from pyspark.sql import DataFrame


class SecondWatermarkAggregator(RealTimePredictor):
    def __init__(self):
        super().__init__()
    
    def parse_df(self, df: DataFrame) -> DataFrame:
        key_schema = StructType([
            StructField("timewindow", 
                StructType([
                    StructField("start", TimestampType(), False),
                    StructField("end", TimestampType(), False)
                ]))
            ])
        value_schema = ArrayType(ArrayType(ArrayType(IntegerType(), False), False), False)

        return df \
            .select(
                F.from_json(F.col("key").cast("string"), schema=key_schema).alias("parsed_key"),
                F.from_json(F.col("value").cast("string"), schema=value_schema).alias("3D_mat")
            ) \
            .withColumns({
                "timewindow": F.col("parsed_key.timewindow")
            }) \
            .drop("parsed_key")

    def to_4d_np(self, df: DataFrame) -> DataFrame:  
        return df \
            .withColumn("datetime", F.col("timewindow.start")) \
            .withWatermark("datetime", f"{self.timewindow*self.history_len} second") \
            .dropDuplicates(["datetime"]) \
            .groupBy(
                F.window(
                    F.col("datetime"), 
                    f"{self.timewindow*self.history_len} second",
                    f"{self.timewindow} second"
                ).alias("timewindow")
            ) \
            .agg(
                F.sort_array(F.collect_list(F.struct("timewindow", "3D_mat"))).alias("4D_mat"),
            ) \
            .withColumns({
                "sorted_4D_mat": F.col("4D_mat.3D_mat")
            }) \
            .select(
                F.col("timewindow").alias("based_timewindow"),
                F.col("4D_mat.timewindow").alias("timewindows"), 
                F.col("sorted_4D_mat") # (history_len, num_lanes, num_sections, num_features)
            )
    
    def real_time_prediction(self, df: DataFrame) -> DataFrame:
        def prediction(input):
            model_name = f'mdl_model_{self.with_ramp_sign}_{self.timewindow}_{self.num_sections}_{self.history_len}_{self.num_features}_{self.num_skip}'
            #model = keras.saving.load_model(f"mdl/models/{model_name}.keras")
            self.model.load_weights(f"training/models/{model_name}.weights.h5")

            if len(input) != self.history_len: # if there is not enough received data to predict
                return np.zeros((self.num_lanes, self.num_sections), dtype=int).tolist()
            
            input_mat = np.reshape(np.array(input), (1, self.history_len, self.num_lanes, self.num_sections, self.num_features))
            pred = self.model.predict(input_mat)
            round_pred = np.rint(pred).astype(int)
            return round_pred.reshape(self.num_lanes, self.num_sections).tolist()

        pred_udf = F.udf(prediction, ArrayType(ArrayType(IntegerType())))

        return df \
            .withColumns({
                "prediction": pred_udf(F.col("sorted_4D_mat")),
                "Global_Time": F.unix_timestamp(F.to_utc_timestamp(F.col("based_timewindow.start"), 'America/Los_Angeles')) + 3600000
            }) \
            .select(F.col("based_timewindow"), F.col("Global_Time"), F.col("prediction")) \
        
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
        
        parsed_df = self.parse_df(df)
        test_df = self.to_4d_np(parsed_df)
        pred_df = self.real_time_prediction(test_df)

        query = pred_df \
            .writeStream \
            .outputMode("append") \
            .format("json") \
            .option('path', 'realtime_pred_res') \
            .option("checkpointLocation", "checkpoints/SecondWatermarkAggregator") \
            .trigger(processingTime='1 minute') \
            .start()

        query.awaitTermination()
        

SecondWatermarkAggregator().init_job()