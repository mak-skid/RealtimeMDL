from shapely.geometry import Polygon
from pyspark.sql import DataFrame, SparkSession

class STManager:
	def aggregate_st_dfs(
			spark: SparkSession, 
			dataset1, 
			dataset2, 
			geometry1, 
			geometry2, 
			id1, 
			id2, 
			geo_relationship, 
			columns_to_aggregate, 
			column_aggregatioin_types, 
			column_alias_list = None
			):
		'''
		Joins two geo-datasets based on spatial relationships such as contains, intersects, touches, etc.
		For each polygon in dataset1, it finds those tuples from dataset2 which satisfy the geo_relationship with the corresponding polygon in dataset1.\
		Those tuples from dataset2 are aggregated using aggregation types such as sum, count, avg to generate a tuple of feature for a polygon in dataset1.
		The dataset from which features need to be aggregated should be dataset2

		Parameters
		...........
		dataset1: pyspark dataframe containing polygon objects
		dataset2: pyspark dataframe which contains the features that need to be aggregated
		geometry1: column name in dataset1 dataframe that contains geometry coordinates
		geometry2: column name in dataset2 dataframe that contains geometry coordinates
		id1: column name in dataset1 dataframe that contains ids of polygons
		id2: column name in dataset2 dataframe that contains ids of temporal steps
		geo_relationship: stands for the type of spatial relationship. It takes 4 different values:
							  SpatialRelationshipType.CONTAINS: geometry in dataset1 completely contains geometry in dataset2
							  SpatialRelationshipType.INTERSECTS: geometry in dataset1 intersects geometry in dataset2
							  SpatialRelationshipType.TOUCHES: geometry in dataset1 touches geometry in dataset2
							  SpatialRelationshipType.WITHIN: geometry in dataset1 in completely within the geometry in dataset2
		columns_to_aggregate: a python list containing the names of columns from dataset2 which need to be aggregated
		column_aggregatioin_types: stands for the type of column aggregations such as sum, count, avg. It takes 5 different values:
								   AggregationType.COUNT: similar to count aggregation type in SQL
								   AggregationType.SUM: similar to sum aggregation type in SQL
								   AggregationType.AVG: similar to avg aggregation type in SQL
								   AggregationType.MIN: similar to min aggregation type in SQL
								   AggregationType.MAX: similar to max aggregation type in SQL
		column_alias_list: Optional, if you want to rename the aggregated columns from the list columns_to_aggregate, provide a list of new names

		Returns
		.......
		a pyspark dataframe consisting of polygon ids from dataset1 and aggregated features from dataset2
		'''

		def __get_columns_selection__():
			expr = ""
			for i in range(len(columns_to_aggregate)):
				if column_aggregatioin_types is not None:
					aggregation = column_aggregatioin_types[i].value
				else:
					aggregation = "COUNT"
				if i != 0:
					expr += ", "
				expr += aggregation + "(" + columns_to_aggregate[i] + ")"
				if column_alias_list is not None:
					expr += " AS " + column_alias_list[i]
			return expr

		select_expr = __get_columns_selection__()

		dataset1.createOrReplaceTempView("dataset1")
		dataset2.createOrReplaceTempView("dataset2")
		dfJoined = spark.sql("SELECT * FROM dataset1 AS d1 INNER JOIN dataset2 AS d2 ON {0}(d1.{1}, d2.{2})".format(geo_relationship.value, geometry1, geometry2))
		dfJoined.createOrReplaceTempView("dfJoined")
		dfJoined = spark.sql("SELECT dfJoined.{0}, dfJoined.{1}, {2} FROM dfJoined GROUP BY dfJoined.{0}, dfJoined.{1} ORDER BY dfJoined.{0}, dfJoined.{1}".format(id2, id1, select_expr))
		return dfJoined