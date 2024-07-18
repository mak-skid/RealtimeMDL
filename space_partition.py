from shapely.geometry import Polygon
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
from sedona.sql.types import GeometryType

class SpacePartition:
	def generate_grid_cells(
		spark: SparkSession, 
		geo_df: DataFrame, 
		geometry: str, 
		partitions_x: int, 
		partitions_y: int
		):
		'''
		Function generates a grid of partitions_x times partitions_y cells
		partitions_x cells along the latitude and partitions_y cells along the longitude

		Parameters
		...........
		geo_df: pyspark dataframe containing a column of geometry type
		geometry: name of the geometry typed column in geo_df dataframe
		partitions_x: number of partitions along latitude
		partitions_y: number of partitions along longitude

		Returns
		........
		a pyspark dataframe constsiting of two columns: id of each cell and polygon object representing each cell
		'''

		geo_df.createOrReplaceTempView("geo_df")
		boundary = spark.sql(
			"SELECT ST_Envelope_Aggr(geo_df.{0}) as boundary FROM geo_df".format(geometry)
			).collect()[0][0]
		x_arr, y_arr = boundary.exterior.coords.xy
		x = list(x_arr)
		y = list(y_arr)

		min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)
		interval_x = (max_x - min_x)/partitions_x
		interval_y = (max_y - min_y)/partitions_y

		polygons = []
		ids = []
		for i in range(partitions_y):
			for j in range(partitions_x):
				polygons.append(
					Polygon(
						[
							[min_x + interval_x * j, min_y + interval_y * i], 
							[min_x + interval_x * (j + 1), min_y + interval_y * i], 
							[min_x + interval_x * (j + 1), min_y + interval_y * (i + 1)], 
							[min_x + interval_x * j, min_y + interval_y * (i + 1)], 
							[min_x + interval_x * j, min_y + interval_y * i]
						]
					)
				)
				ids.append(i*partitions_x + j)

		schema_cells = StructType(
			[
			StructField("cell_id", IntegerType(), False),
			StructField("geometry", GeometryType(), False)
			])

		return spark.createDataFrame(zip(ids, polygons), schema = schema_cells)

