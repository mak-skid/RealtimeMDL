"""
Functions to visualise the US101 dataset

Author: Makoto Ono
"""

from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from utils.datapreprocessing_utils import create_np_matrices
from pyspark.sql import DataFrame, functions as F
from matplotlib.collections import LineCollection
import geopandas as gpd


def us101_section_vis(df: DataFrame, num_section_split: int, timestamp: str, with_ramp: bool) -> None:    
    lanes = ["1", "2", "3", "4", "5", "6 (Ramp)"] if with_ramp else ["1", "2", "3", "4", "5"]
    sections = [i for i in range(num_section_split+1)]

    vel_matrix, dens_matrix, acc_matrix = create_np_matrices(df, num_lanes=len(lanes), num_sections=num_section_split+1, with_ramp=with_ramp)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    im1 = ax1.imshow(vel_matrix, cmap='turbo_r', norm=Normalize(vmin=0, vmax=60))
    im2 = ax2.imshow(dens_matrix, cmap='turbo', norm=Normalize(vmin=0, vmax=150))
    #im3 = ax3.imshow(acc_matrix, cmap='turbo_r', norm=Normalize(vmin=-3, vmax=3))

    ax1.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax1.set_xticks(np.arange(len(sections)), labels=sections)

    ax2.set_yticks(np.arange(len(lanes)), labels=lanes)
    ax2.set_xticks(np.arange(len(sections)), labels=sections)

    #ax3.set_yticks(np.arange(len(lanes)), labels=lanes)
    #ax3.set_xticks(np.arange(len(sections)), labels=sections)

    plt.setp(ax1.get_yticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.setp(ax2.get_yticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")        
    #plt.setp(ax3.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor") 

    for i in range(len(lanes)): # y axis
        for j in range(len(sections)): # x axis
            text = ax1.text(j, i, vel_matrix[i][j],
                        ha="center", va="center", color="w")
            text = ax2.text(j, i, int(dens_matrix[i][j]),
                            ha="center", va="center", color="w")
            #text = ax3.text(j, i, acc_matrix[i][j], ha="center", va="center", color="w")

    ax1.set_title(f"Average Velocity (mph) by Section and Lane at t={timestamp}")
    ax2.set_title(f"Density by Section and Lane at t={timestamp}")
    #ax3.set_title(f"Average acceleration by Section and Lane at t={timestamp}")

    plt.subplots_adjust(hspace=0.4)
    plt.show()


def visualise_shockwave(df: DataFrame, lane_id: int, max_dist: int, max_elapsed_time: int) -> None:
    """
    df: pyspark dataframe containing the dataset filtered by lane_id
    lane_id: int, lane id used to filter the dataframe
    max_dist: int, maximum distance in feet
    max_elapsed_time: int, maximum elapsed time in ms
    """

    pdf = df.toPandas()

    vehicles = pdf.groupby("Vehicle_ID")

    fig, ax = plt.subplots()
    norm = plt.Normalize(0, 60)

    for _, points in vehicles:
        x = points["ElapsedTime"].to_numpy()
        y = points["Distance"].to_numpy()
        cols = points["v_Vel"].to_numpy()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='rainbow_r', norm=norm)
        lc.set_array(cols)
        lc.set_linewidth(1)
        ax.add_collection(lc)
        
    fig.colorbar(lc, ax=ax, label="Velocity (mph)")
    ax.set_xlim(0, max_elapsed_time)
    ax.set_ylim(0, max_dist)

    ax.set_xlabel("Time Elapsed (ms)")
    ax.set_ylabel("Distance (ft)")
    ax.set_title(f"Lane {lane_id}")

    plt.show()


def granular_snapshot(df: DataFrame, timestamp_ms: int) -> None:
    """
    df: pyspark dataframe containing the US101 dataset with gps_geom column
    timestamp_ms: int, timestamp in milliseconds
    """
    snapshot_df = df \
        .select("Location", "ElapsedTime", "Distance", "Vehicle_ID", "Lane_ID", "v_Vel", "v_Acc", "gps_geom") \
        .filter(
            (F.col("ElapsedTime") == timestamp_ms)
        ) \
        .sort("ElapsedTime")

    pdf_snapshot = snapshot_df.toPandas()
    gdf_snapshot = gpd.GeoDataFrame(pdf_snapshot, geometry="gps_geom")

    gdf_snapshot.plot(
        figsize=(15, 15),
        column="v_Vel",
        cmap="RdBu",
        markersize=0.8,
        legend=True
    )

