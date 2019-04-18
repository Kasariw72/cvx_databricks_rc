# Databricks notebook source
# MAGIC %md # Notebook #2
# MAGIC 
# MAGIC In this notebook, we worked with the result dataset from Notebook #1 and computed rolling statistics (mean, difference, std, max, min) for a list of features over various time windows.  
# MAGIC This was the most time consuming and computational expensive part of the entire tutorial. We encountered some roadblocks and found some workarounds. Please see below for more details.
# MAGIC 
# MAGIC ## Outline
# MAGIC 
# MAGIC - [Define Rolling Features and Window Sizes](#Define-list-of-features-for-rolling-compute,-window-sizes)
# MAGIC - [Issues and Solutions](#What-issues-we-encountered-using-Pyspark-and-how-we-solved-them?)
# MAGIC - [Rolling Compute](#Rolling-Compute)
# MAGIC   - [Rolling Mean](#Rolling-Mean)
# MAGIC   - [Rolling Difference](#Rolling-Difference)
# MAGIC   - [Rolling Std](#Rolling-Std)
# MAGIC   - [Rolling Max](#Rolling-Max)
# MAGIC   - [Rolling Min](#Rolling-Min)
# MAGIC - [Join Results](#Join-result-dataset-from-the-five-rolling-compute-cells:)
# MAGIC   

# COMMAND ----------

import pyspark.sql.functions as F
import time
import subprocess
import sys
import os
import re

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col,udf,lag,date_add,explode,lit,concat,unix_timestamp
from pyspark.sql.dataframe import *
from pyspark.sql.window import Window
from pyspark.sql.types import DateType
from datetime import datetime, timedelta
from pyspark.sql import Row

start_time = time.time()


# COMMAND ----------

# MAGIC %md ## Define list of features for rolling compute, window sizes

# COMMAND ----------

rolling_features = [
    'warn_type1_total', 'warn_type2_total', 
    'pca_1_warn','pca_2_warn', 'pca_3_warn', 'pca_4_warn', 'pca_5_warn',
    'pca_6_warn','pca_7_warn', 'pca_8_warn', 'pca_9_warn', 'pca_10_warn',
    'pca_11_warn','pca_12_warn', 'pca_13_warn', 'pca_14_warn', 'pca_15_warn',
    'pca_16_warn','pca_17_warn', 'pca_18_warn', 'pca_19_warn', 'pca_20_warn',
    'problem_type_1', 'problem_type_2', 'problem_type_3','problem_type_4',
    'problem_type_1_per_usage1','problem_type_2_per_usage1',
    'problem_type_3_per_usage1','problem_type_4_per_usage1',
    'problem_type_1_per_usage2','problem_type_2_per_usage2',
    'problem_type_3_per_usage2','problem_type_4_per_usage2',                
    'fault_code_type_1_count', 'fault_code_type_2_count', 'fault_code_type_3_count', 'fault_code_type_4_count',                          
    'fault_code_type_1_count_per_usage1','fault_code_type_2_count_per_usage1',
    'fault_code_type_3_count_per_usage1', 'fault_code_type_4_count_per_usage1',
    'fault_code_type_1_count_per_usage2','fault_code_type_2_count_per_usage2',
    'fault_code_type_3_count_per_usage2', 'fault_code_type_4_count_per_usage2']
               
# lag window 3, 7, 14, 30, 90 days
lags = [3, 7, 14, 30, 90]

print(len(rolling_features))


# COMMAND ----------

# MAGIC %md ## What issues we encountered using Pyspark and how we solved them?
# MAGIC 
# MAGIC -  If the entire list of **46 features** and **5 time windows** were computed for **5 different types of rolling** (mean, difference, std, max, min) all in one go, we always ran into "StackOverFlow" error. 
# MAGIC -  It was because the lineage was too long and Spark could not handle it.
# MAGIC -  We could either create checkPoint and materialize it throughout the process.
# MAGIC -  OR break the workload into chunks and save the result from each chunk as parquet file.
# MAGIC 
# MAGIC ## A few things we found helpful:
# MAGIC -  Before the rolling compute, save the upstream work as a parquet file in Notebook_1 ("Notebook_1_DataCleansing_FeatureEngineering"). It will speed up the whole process because we no need to repeat all the previous steps. It will also help reduce the lineage.
# MAGIC -  Print out the lag and feature name to track progress.
# MAGIC -  Use "htop" command from the terminal to keep track how many CPUs are running for a particular task. For rolling compute, we were considering two potential approaches: 1) Use Spark clusters on HDInsight to perform rolling compute in parallel; 2) Use single node Spark on a powerful VM. By looking at htop dashboard, we saw all the 32 cores were running at the same time for a single task (for example compute rolling mean). So if say we divide the workload onto multiple nodes and each node runs a type of rolling compute, the amount of time taken will be comparable with running everything in a sequential manner on a single node Spark on a powerful machine.
# MAGIC -  Use "%%time" for each cell to get an estimate of the total run time, we will then have a better idea where and what to optimze the process.
# MAGIC -  Materialize the intermediate results by either caching in memory or writing as parquet files. We chose to save as parquet files because we did not want to repeat the compute again in case cache() did not work or any part of the rolling compute did not work.
# MAGIC -  Why parquet? There are many reasons, just to name a few: parquet not only saves the data but also the schema, it is a preferred file format by Spark, you are allowed to read only the data you need, etc..
# MAGIC 
# MAGIC <br>

# COMMAND ----------

# MAGIC %md ## Rolling Compute
# MAGIC ### Rolling Mean

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # Load result dataset from Notebook #1
# MAGIC df = sqlContext.read.parquet('/mnt/resource/PysparkExample/notebook1_result.parquet')
# MAGIC 
# MAGIC for lag_n in lags:
# MAGIC     wSpec = Window.partitionBy('deviceid').orderBy('date').rowsBetween(1-lag_n, 0)
# MAGIC     for col_name in rolling_features:
# MAGIC         df = df.withColumn(col_name+'_rollingmean_'+str(lag_n), F.avg(col(col_name)).over(wSpec))
# MAGIC         print("Lag = %d, Column = %s" % (lag_n, col_name))
# MAGIC 
# MAGIC # Save the intermediate result for downstream work
# MAGIC df.write.mode('overwrite').parquet('/mnt/resource/PysparkExample/data_rollingmean.parquet')

# COMMAND ----------

# MAGIC %md ### Rolling Difference

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # Load result dataset from Notebook #1
# MAGIC df = sqlContext.read.parquet('/mnt/resource/PysparkExample/notebook1_result.parquet')
# MAGIC 
# MAGIC for lag_n in lags:
# MAGIC     wSpec = Window.partitionBy('deviceid').orderBy('date').rowsBetween(1-lag_n, 0)
# MAGIC     for col_name in rolling_features:
# MAGIC         df = df.withColumn(col_name+'_rollingdiff_'+str(lag_n), col(col_name)-F.avg(col(col_name)).over(wSpec))
# MAGIC         print("Lag = %d, Column = %s" % (lag_n, col_name))
# MAGIC 
# MAGIC rollingdiff = df.select(['key'] + list(s for s in df.columns if "rollingdiff" in s))
# MAGIC 
# MAGIC # Save the intermediate result for downstream work
# MAGIC rollingdiff.write.mode('overwrite').parquet('/mnt/resource/PysparkExample/rollingdiff.parquet')

# COMMAND ----------

# MAGIC %md ### Rolling Std

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # Load result dataset from Notebook #1
# MAGIC df = sqlContext.read.parquet('/mnt/resource/PysparkExample/notebook1_result.parquet')
# MAGIC 
# MAGIC for lag_n in lags:
# MAGIC     wSpec = Window.partitionBy('deviceid').orderBy('date').rowsBetween(1-lag_n, 0)
# MAGIC     for col_name in rolling_features:
# MAGIC         df = df.withColumn(col_name+'_rollingstd_'+str(lag_n), F.stddev(col(col_name)).over(wSpec))
# MAGIC         print("Lag = %d, Column = %s" % (lag_n, col_name))
# MAGIC 
# MAGIC # There are some missing values for rollingstd features
# MAGIC rollingstd_features = list(s for s in df.columns if "rollingstd" in s)
# MAGIC df = df.fillna(0, subset=rollingstd_features)
# MAGIC rollingstd = df.select(['key'] + list(s for s in df.columns if "rollingstd" in s))
# MAGIC 
# MAGIC # Save the intermediate result for downstream work
# MAGIC rollingstd.write.mode('overwrite').parquet('/mnt/resource/PysparkExample/rollingstd.parquet')

# COMMAND ----------

# MAGIC %md ### Rolling Max

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # Load result dataset from Notebook #1
# MAGIC df = sqlContext.read.parquet('/mnt/resource/PysparkExample/notebook1_result.parquet')
# MAGIC 
# MAGIC for lag_n in lags:
# MAGIC     wSpec = Window.partitionBy('deviceid').orderBy('date').rowsBetween(1-lag_n, 0)
# MAGIC     for col_name in rolling_features:
# MAGIC         df = df.withColumn(col_name+'_rollingmax_'+str(lag_n), F.max(col(col_name)).over(wSpec))
# MAGIC         print("Lag = %d, Column = %s" % (lag_n, col_name))
# MAGIC 
# MAGIC rollingmax = df.select(['key'] + list(s for s in df.columns if "rollingmax" in s))
# MAGIC 
# MAGIC # Save the intermediate result for downstream work
# MAGIC rollingmax.write.mode('overwrite').parquet('/mnt/resource/PysparkExample/rollingmax.parquet')

# COMMAND ----------

# MAGIC %md ### Rolling Min

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # Load result dataset from Notebook #1
# MAGIC df = sqlContext.read.parquet('/mnt/resource/PysparkExample/notebook1_result.parquet')
# MAGIC 
# MAGIC for lag_n in lags:
# MAGIC     wSpec = Window.partitionBy('deviceid').orderBy('date').rowsBetween(1-lag_n, 0)
# MAGIC     for col_name in rolling_features:
# MAGIC         df = df.withColumn(col_name+'_rollingmin_'+str(lag_n), F.min(col(col_name)).over(wSpec))
# MAGIC         print("Lag = %d, Column = %s" % (lag_n, col_name))
# MAGIC 
# MAGIC rollingmin = df.select(['key'] + list(s for s in df.columns if "rollingmin" in s))
# MAGIC 
# MAGIC # Save the intermediate result for downstream work
# MAGIC rollingmin.write.mode('overwrite').parquet('/mnt/resource/PysparkExample/rollingmin.parquet')

# COMMAND ----------

# MAGIC %md ## Join result dataset from the five rolling compute cells:
# MAGIC -  Join in Spark is usually very slow, it is better to reduce the number of partitions before the join.
# MAGIC -  Check the number of partitions of the pyspark dataframe.
# MAGIC -  **repartition vs coalesce**. If we only want to reduce the number of partitions, it is better to use coalesce because repartition involves reshuffling which is computational more expensive and takes more time.
# MAGIC <br>

# COMMAND ----------

# Import result dataset 
rollingmean = sqlContext.read.parquet('/mnt/resource/PysparkExample/data_rollingmean.parquet')
rollingdiff = sqlContext.read.parquet('/mnt/resource/PysparkExample/rollingdiff.parquet')
rollingstd = sqlContext.read.parquet('/mnt/resource/PysparkExample/rollingstd.parquet')
rollingmax = sqlContext.read.parquet('/mnt/resource/PysparkExample/rollingmax.parquet')
rollingmin = sqlContext.read.parquet('/mnt/resource/PysparkExample/rollingmin.parquet')

# Check the number of partitions for each dataset
print(rollingmean.rdd.getNumPartitions())
print(rollingdiff.rdd.getNumPartitions())
print(rollingstd.rdd.getNumPartitions())
print(rollingmax.rdd.getNumPartitions())
print(rollingmin.rdd.getNumPartitions())


# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # To make join faster, reduce the number of partitions (not necessarily to "1")
# MAGIC rollingmean = rollingmean.coalesce(1)
# MAGIC rollingdiff = rollingdiff.coalesce(1)
# MAGIC rollingstd = rollingstd.coalesce(1)
# MAGIC rollingmax = rollingmax.coalesce(1)
# MAGIC rollingmin = rollingmin.coalesce(1)
# MAGIC 
# MAGIC rolling_result = rollingmean.join(rollingdiff, 'key', 'inner')\
# MAGIC                  .join(rollingstd, 'key', 'inner')\
# MAGIC                  .join(rollingmax, 'key', 'inner')\
# MAGIC                  .join(rollingmin, 'key', 'inner')
# MAGIC             
# MAGIC 
# MAGIC ## Write the final result as parquet file for downstream work in Notebook_3
# MAGIC rolling_result.write.mode('overwrite').parquet('/mnt/resource/PysparkExample/notebook2_result.parquet')