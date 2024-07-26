# Convert the coordinate system
from pyspark import RDD
from sedona.sql.st_functions import ST_Transform, ST_Y, ST_X
from sedona.sql.st_constructors import ST_Point
from pyspark.sql import functions as F
from pyspark.sql import SparkSession, DataFrame
import numpy as np
import torch

def convert_coordinate_system(df: DataFrame) -> DataFrame:
    """
    df: pyspark dataframe containing the US101 dataset
    
    Returns
    -------
    df: pyspark dataframe containing the converted coordinate system
    """
    df = df \
        .withColumn("geometry", ST_Point(df["Global_X"], df["Global_Y"])) \
        .withColumn("gps_geom", ST_Transform("geometry", F.lit("EPSG:2227"), F.lit("EPSG:4326"))) \
        .drop("Global_X", "Global_Y", "geometry") \
        .withColumns({
            "lat": ST_Y("gps_geom"),
            "lon": ST_X("gps_geom")
        })
    return df

def convert_timestamp(df: DataFrame) -> DataFrame:
    """
    df: pyspark dataframe containing the US101 dataset
    
    Returns
    -------
    df: pyspark dataframe containing the datetime column converted from utc_timestamp
    """
    return df.withColumn("datetime", 
                   F.from_utc_timestamp(
                       F.timestamp_millis(
                           F.col("Global_Time") - 3600000 # before 2006, there was no daylight saving time so we need to subtract 1 hour here
                        ),
                    'America/Los_Angeles'))

def add_distance_and_time_cols(df: DataFrame) -> DataFrame:
    """
    df: pyspark dataframe containing the US101 dataset

    Returns
    -------
    df: pyspark dataframe containing the data of Distance and ElapsedTime
    """
    return df.withColumns({
            "Distance": F.sqrt(F.pow(F.col("Local_X"), 2) + F.pow(F.col("Local_Y"), 2)),
            "ElapsedTime": F.col("Global_Time") - 1113433135300
        })

def us101_filter(df: DataFrame) -> DataFrame:
    return df \
        .filter(F.col("Location") == "us-101") \
        .withColumn("Lane_ID", 
                    F.when(
                        F.col("Lane_ID").isin([7, 8]), 6) # merge lanes 7 and 8 into 6
                        .otherwise(F.col("Lane_ID"))
                        )

def hour_filter(df: DataFrame, location: str, hour: list) -> DataFrame:
    if location == "us-101":
        deduction = 5413844400

    filtered_df= df.filter((F.col("hour").isin(hour))).sort("datetime") \
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
            .withColumn("ElapsedTime", F.col("ElapsedTime") - deduction) \
            .sort("ElapsedTime")
        
    print(f"{location} {hour}h Data Filtered")
    return filtered_df

def lane_filter(df: DataFrame, lane_id: int) -> DataFrame:
    """
    df: pyspark dataframe containing the US101 dataset
    lane_id: int, lane id to filter the dataframe

    Returns
    -------
    df: pyspark dataframe containing the data of Location, ElapsedTime, hour, 
        Distance, Vehicle_ID, v_Vel, v_Acc for the specified lane_id
    """
    return df \
        .select("Location", "ElapsedTime", "hour", "Distance", "Vehicle_ID", "Lane_ID", "v_Vel", "v_Acc") \
        .filter(
            (F.col("Lane_ID") == lane_id)
        ) \
        .sort("ElapsedTime")

def create_np_matrices(df: DataFrame, num_lanes: int, num_sections: int, with_ramp: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    df: pyspark dataframe containing the US101 dataset
    num_section: int, number of sections to divide the dataset

    Returns
    --------
    matrices: tuple of three 2D numpy arrays (num_lanes, num_section+1) containing the avg(v_Vel), count, and avg(v_Acc) values
    """

    vel_matrix = np.full((num_lanes, num_sections), 60) # fill with 60 mph 
    dens_matrix = np.zeros((num_lanes, num_sections))
    acc_matrix = np.zeros((num_lanes, num_sections))

    # Fill the matrix with the corresponding avg(v_Vel) values
    for row in iter:
        lane_index = row["Lane_ID"]

        # escape when with_ramp is False and lane_id is 6 (lane_index is 5)
        if not with_ramp and lane_index == 6: 
            continue

        section_index = row["Section_ID"]
        avg_vel = row["avg(v_Vel)"]
        count = row["count"]
        avg_acc = row["avg(v_Acc)"]

        if section_index == num_sections:
            section_index = num_sections - 1

        vel_matrix[lane_index-1][section_index] = avg_vel
        dens_matrix[lane_index-1][section_index] = count
        acc_matrix[lane_index-1][section_index] = avg_acc
    
    return vel_matrix, dens_matrix, acc_matrix

    ### IDEA### provide the results which contains the result with and without ramp

def tensor_to_np_matrices(tensor: torch.Tensor) -> np.ndarray:
    """
    tensor: torch tensor containing the data of avg(v_Vel), count, and avg(v_Acc) values

    Returns
    --------
    matrices: 3D numpy array (3, num_lanes, num_section+1) containing the avg(v_Vel), count, and avg(v_Acc) values
    """
    return tensor.cpu().numpy()

def rdd_to_np_matrices(key: int, iter, num_lanes: int, num_sections: int, with_ramp: bool = True) -> tuple[int, np.ndarray]:
    """
    iter: RDD iterator containing the US101 dataset
    num_lanes: int, number of lanes in the dataset
    num_sections: int, number of sections to divide the dataset
    with_ramp: bool, whether to include the ramp lane or not

    Returns
    --------
    matrices: 3D numpy array (3, num_lanes, num_section+1) containing the avg(v_Vel), count, and avg(v_Acc) values
    """

    # Create an empty matrix with the dimensions of Section_ID and Lane_ID
    vel_matrix = np.full((num_lanes, num_sections), 60) # fill with 60 mph 
    dens_matrix = np.zeros((num_lanes, num_sections))
    acc_matrix = np.zeros((num_lanes, num_sections))

    # Fill the matrix with the corresponding avg(v_Vel) values
    for row in iter:
        lane_index = row["Lane_ID"]

        # escape when with_ramp is False and lane_id is 6 (lane_index is 5)
        if not with_ramp and lane_index == 6: 
            continue

        section_index = row["Section_ID"]
        avg_vel = row["avg(v_Vel)"]
        count = row["count"]
        avg_acc = row["avg(v_Acc)"]

        if section_index == num_sections:
            section_index = num_sections - 1

        vel_matrix[lane_index-1][section_index] = avg_vel
        dens_matrix[lane_index-1][section_index] = count
        acc_matrix[lane_index-1][section_index] = avg_acc

    matrices = np.stack([vel_matrix, dens_matrix, acc_matrix])
    return key, matrices


def section_agg(df: DataFrame, max_dist: int, num_section_splits: int) -> DataFrame:
    """
    df: pyspark dataframe containing the US101 dataset

    Returns
    --------
    df: pyspark dataframe containing the aggregated values of avg(v_Vel), avg(v_Acc), and count, 
        grouped by ElapsedTime, Section_ID, and Lane_ID
    max_dist: int, maximum distance in the dataset
    num_section_splits: int, number of splits to perform on road section of the dataset
    """
    df = df \
        .withColumn("Section_ID", 
            F.round(
                (F.col("Distance") / F.lit(max_dist // num_section_splits)).cast("integer")
            ) # gives a Section ID to each datapoint 
        ) \
        .select("ElapsedTime", "Lane_ID", "v_Vel", "v_Acc", "Section_ID") \
        .groupBy("ElapsedTime", "Section_ID", "Lane_ID") \
        .agg(
            F.round(F.avg("v_Vel"), 1).alias("avg(v_Vel)"), 
            F.round(F.avg("v_Acc"), 2).alias("avg(v_Acc)"), 
            F.count("*").alias("count")
        )
    print("Section Aggregation Sample Result: ")
    df.show(1)
    return df

def timewindow_agg(df: DataFrame, start: int, end: int, timewindow: int) -> DataFrame:
    """
    df: pyspark dataframe containing the US101 dataset
    start: int, start time in seconds
    end: int, end time in seconds
    timewindow: int, time window in seconds

    Returns
    -------
    df: pyspark dataframe containing the aggregated values of avg(v_Vel), avg(v_Acc), and count 
        within the timewindow, grouped by TimeWindow, Section_ID, and Lane_ID
    """
    df = df \
        .filter((F.col("ElapsedTime") >= start * 1000) & (F.col("ElapsedTime") < end * 1000 - 45)) \
        .withColumn("TimeWindow",                                                       # subtract 45 seconds to remove the last incomplete trajectories
            F.round((F.col("ElapsedTime") / F.lit(timewindow * 1000)).cast("integer")) # gives a TimeWindow ID of every 30 sec to each datapoint 
        ) \
        .groupBy("TimeWindow", "Section_ID", "Lane_ID") \
        .agg(
            F.round(F.avg("avg(v_Vel)"), 1).alias("avg(v_Vel)"), 
            F.round(F.avg("avg(v_Acc)"), 2).alias("avg(v_Acc)"), 
            F.count("*").alias("count")
        )
    print("Time Window Aggregation Sample Result: ")
    df.show(1)
    return df

def add_timewindow_col(df: DataFrame, start: int, end: int, timewindow: int) -> DataFrame:
    return df \
        .filter((F.col("ElapsedTime") >= start * 1000) & (F.col("ElapsedTime") < end * 1000)) \
        .withColumn("TimeWindow", 
            F.round((F.col("ElapsedTime") / F.lit(timewindow * 1000)).cast("integer")) # gives a TimeWindow ID of every n sec to each datapoint 
        )