from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def change_columns_names(df, symbol, after_group_by=False):
    columns = df.columns

    if not after_group_by:

        for i, column in enumerate(columns):
            df = df.withColumnRenamed(column, f"{i + 1}_{symbol}")

    else:
        for i, column in enumerate(columns):
            if i == 0:
                continue
            df = df.withColumnRenamed(column, f"{i}_{symbol}")

    return df


def add_row_num(df, col_name):
    df = df.withColumn("temp", lit("T"))
    w = Window().partitionBy("temp").orderBy(lit('T'))

    df = df.withColumn(col_name, row_number().over(w)).drop("temp")

    return df


def get_dist_expr(columns_lst_1, columns_lst_2):
    n = len(columns_lst_1)
    dist_expr = "("

    for i in range(n):
        dist_expr += f"POWER({columns_lst_1[i]} - {columns_lst_2[i]}, 2)"

        if i == n - 1:
            dist_expr += ") as dist"

        else:
            dist_expr += " + "

    return dist_expr


def kmeans_fit(data, k, max_iter, q, init):
    spark = SparkSession.builder.getOrCreate()
    
    temp = spark.createDataFrame(init)
    points = change_columns_names(data, symbol='p')
    centroids = change_columns_names(temp, symbol='c')

    points_columns = list(points.columns)
    centroids_columns = list(centroids.columns)
    
    # expr for the selectExpr function
    # we used this function to compute the distance
    # we are computing the norm square (not taking sqrt) in order to minimize computations and therefore to save time
    # since norm is a non-negative function, by the power rules, the total order stays the same
    expr = points_columns + centroids_columns
    dist_expr = get_dist_expr(points_columns, centroids_columns)
    expr += [dist_expr, "point_num", "cluster_num"]
    
    # adding a number for each point in order to distinguish between them,
    # as the data may contains duplicates of the same point
    points = add_row_num(points, col_name="point_num")
    # adding a number for each centroid in order to represent each cluster
    centroids = add_row_num(centroids, col_name="cluster_num")
    
    # rearrange the columns positions
    centroids = centroids.select(*(["cluster_num"] + centroids_columns))

    iter_num = 1

    while iter_num <= max_iter:
        
        # cross join between the points df and the centroids df,
        # in order the copmute the distance from each point to each centroid
        points_centroids = points.crossJoin(centroids)
        points_centroids_dist = points_centroids.selectExpr(expr)\
                                                .drop(*centroids_columns)

        # finds the centroid (and therefore the cluster) that is the closest to each point
        w = Window().partitionBy("point_num").orderBy("dist")
        points_clusters_dist = points_centroids_dist.withColumn("dist_rank", row_number().over(w))\
                                                    .filter("dist_rank == 1")\
                                                    .drop("dist_rank", "point_num")
        
        # for each cluster, sort its points by their distance to the centorid (ascending order), and filter each q-th point
        w = Window().partitionBy("cluster_num").orderBy("dist")
        relevent_points = points_clusters_dist.withColumn("point_rank", row_number().over(w))\
                                              .drop("dist")\
                                              .filter(f"point_rank%{q} != 0")\
                                              .drop("point_rank")
        
        # compute the new centroids with the points from the previous step
        temp_new_centroids = relevent_points.groupby("cluster_num")\
                                            .avg(*points_columns)\
                                            .sort("cluster_num")
                                            
        new_centroids = change_columns_names(temp_new_centroids, symbol='c', after_group_by=True)
        
        # checks if the centroids changed
        diff_lst = new_centroids.exceptAll(centroids).take(1)

        if len(diff_lst) == 0:
            break
        
        # using 'take' function was faster then 'collect' function
        centroids_list = new_centroids.take(k)
        centroids = spark.createDataFrame(centroids_list)

        iter_num += 1

    return centroids
