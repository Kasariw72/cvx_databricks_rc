# Databricks notebook source
# MAGIC %md # Notebook #1
# MAGIC In this notebook, we show how to import, clean and create features relevent for predictive maintenance data using PySpark. This notebook uses Spark **2.0.2** and Python Python **2.7.5**. The API documentation for that version can be found
# MAGIC 
# MAGIC ## Outline
# MAGIC 
# MAGIC - [Import Data](#Import-Data)
# MAGIC - [Data Exploration & Cleansing](#Data-Exploration-&-Cleansing)
# MAGIC - [Feature Engineering](#Feature-Engineering)
# MAGIC - [Save Result](#Save-Result)

# COMMAND ----------

import subprocess
import sys
import os
import re
import time
import atexit
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.functions import concat, col, udf, lag, date_add, explode, lit, unix_timestamp
from pyspark.sql.functions import month, weekofyear, dayofmonth
from pyspark.sql.types import *
from pyspark.sql.types import DateType
from pyspark.sql.dataframe import *
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.ml.classification import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml.feature import StandardScaler, PCA, RFormula
from pyspark.ml import Pipeline, PipelineModel

# data transforamtion and manipulation
import pandas as pd
#import pandas_datareader.data as web
import numpy as np
# prevent crazy long pandas prints
pd.options.display.max_columns = 16
pd.options.display.max_rows = 16
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)

# basic functionalities
import datetime
import itertools
import math

start_time = time.time()

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC %config InlineBackend.figure_format = 'retina'
# MAGIC 
# MAGIC plt.style.use('seaborn')
# MAGIC sns.set_style("whitegrid", {'axes.grid' : False})
# MAGIC #set_matplotlib_formats('pdf', 'png')
# MAGIC plt.rcParams['savefig.dpi'] = 80
# MAGIC plt.rcParams['figure.autolayout'] = False
# MAGIC plt.rcParams['figure.figsize'] = (16, 8)
# MAGIC plt.rcParams['axes.labelsize'] = 16
# MAGIC plt.rcParams['axes.labelweight'] = 'bold'
# MAGIC plt.rcParams['axes.titlesize'] = 20
# MAGIC plt.rcParams['axes.titleweight'] = 'bold'
# MAGIC plt.rcParams['font.size'] = 16
# MAGIC plt.rcParams['lines.linewidth'] = 2.0
# MAGIC plt.rcParams['lines.markersize'] = 8
# MAGIC plt.rcParams['legend.fontsize'] = 14
# MAGIC plt.rcParams['text.usetex'] = False
# MAGIC #plt.rcParams['font.family'] = "serif"
# MAGIC plt.rcParams['font.serif'] = "cm"
# MAGIC plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "/databricks/driver/RC_T_MAIN_DATA_01/"
sparkAppName = "RC_T_MAIN_DATA_01"
hdfsPathCSV = "dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA_0*.csv"

# COMMAND ----------

#3 - Setup SparkSession(SparkSQL)
spark = (SparkSession
         .builder
         .appName(sparkAppName)
         .getOrCreate())
print(spark)

# COMMAND ----------

# MAGIC %md ## Import Data

# COMMAND ----------

# MAGIC %md #### We initially encountered some issue while reading the data
# MAGIC -  For Spark 2.0 and above, we can use the sqlContext.read.csv to import data directly into the Spark context. The data import seems to work fine and you can also perform some data transformation without any problem. However, when we tried to show the top n rows of the entire data (e.g. data.show(3)) or do some data manipulation on certain columns downstream, we encountered error of “Null Pointer” or “NumberFormatException: null”.
# MAGIC -  In our case, it was because for some numeric columns with missing records containing "", Spark still recognizes those column as numeric but somehow cannot parse them correctly. Hopefully the future version of Spark could handle such issue more intelligently. 
# MAGIC -  We fixed the problem by reformating the data before loading into Spark context as csv format.
# MAGIC ```bash
# MAGIC  cat data.csv | tr -d "\"\"" > sampledata.csv 
# MAGIC ```

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/mnt/Exploratory/WCLD/dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transfer datafile from Data Lake to Databrick Driver

# COMMAND ----------

#4 - Read file to spark DataFrame
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark)

df = (sqlContext
        .read
        .option("header","true")
        .option("inferSchema", "true")
        .csv(hdfsPathCSV)
        #.parquet(hdfsPathParquet)
     )
# If the path don't have file:/// -> it will call hdfs instead of local file system
df.cache()
print("finish caching data in Spark!")

# COMMAND ----------

meanDic = {'RESI_LL_LUBE_OIL_PRESS': -13.34679, 'RESI_HH_LUBE_OIL_TEMP': -15.00072, 'RESI_LL_SPEED': 819.3537983, 'RESI_HH_SPEED': -182.47516, 'RESI_LL_VIBRATION': 0.3297389, 'RESI_HH_VIBRATION': -0.2719401, 'RESI_HH_THROW1_DIS_TEMP': 281.231386, 'RESI_HH_THROW1_SUC_TEMP': -79.90926, 'LUBE_OIL_PRESS': 68.853477, 'LUBE_OIL_TEMP': 170.73135, 'THROW_1_DISC_PRESS': 493.11111, 'THROW_1_DISC_TEMP': 252.18327, 'THROW_1_SUC_PRESS': 166.40938, 'THROW_1_SUC_TEMP': 124.16805, 'THROW_2_DISC_PRESS': 1213.7482, 'THROW_2_DISC_TEMP': 268.50851, 'THROW_2_SUC_PRESS': 490.68488, 'THROW_2_SUC_TEMP': 133.93256, 'THROW_3_DISC_PRESS': 470.68506, 'THROW_3_DISC_TEMP': 243.81787, 'THROW_3_SUC_PRESS': 165.38683, 'THROW_3_SUC_TEMP': 109.06955, 'THROW_4_DISC_PRESS': 1210.5085, 'THROW_4_DISC_TEMP': 286.57486, 'VIBRATION': 0.17999946, 'CYL_1_TEMP': 1242.7988, 'CYL_2_TEMP': 1241.1833, 'CYL_3_TEMP': 1314.7606, 'CYL_4_TEMP': 1213.8179, 'CYL_5_TEMP': 1234.2267, 'CYL_6_TEMP': 1247.9746, 'CYL_7_TEMP': 1224.9188, 'CYL_8_TEMP': 1231.4215, 'CYL_9_TEMP': 1222.579, 'CYL_10_TEMP': 1203.325, 'CYL_11_TEMP': 1192.7275, 'CYL_12_TEMP': 1173.8546, 'LUBE_OIL_PRESS_ENGINE': 69.623924, 'MANI_PRESS': 102.42593, 'RIGHT_BANK_EXH_TEMP': 1246.0, 'SPEED': 1057.7806, 'VIBRA_ENGINE': 0.2, 'S1_RECY_VALVE': 20.171038, 'S1_SUCT_PRESS': 168.81329, 'S1_SUCT_TEMPE': 120.17609, 'S2_STAGE_DISC_PRESS': 1198.1665, 'S2_SCRU_SCRUB_LEVEL': 40.054184, 'GAS_LIFT_HEAD_PRESS': 1211.9084, 'IN_CONT_CONT_VALVE': 100.0, 'IN_SEP_PRESS': 175.5134, 'WH_PRESSURE': 370.74503}

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
      #med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
      RULFrom = 60
      RULTo = 10080
      bfEventType = "UCM"
      
      if meanDic=="":
        med = getDataByCol(colName,RULFrom,RULTo,bfEventType)
      else:
        med = meanDic[colName]
        
      #print(:med)
      meanDic[colName] = med
      
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(DoubleType()))
      pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(med))
      
      print("Relaced [",colName,"] by ", med)
    
    return pySparkDF

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
def convertStr2Int(pySparkDF, columnList):
    for colName in columnList:
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(IntegerType()))    
    return pySparkDF

def getDataByCol(colName,RULFrom,RULTo,bfEventType):
  #sql = "select "+colName+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and BF_SD_TYPE='"+bfEventType+"' and "+colName+"<>'NULL' and "+colName+" is not null"
  sql = "select "+colName+" from rc_data where RUL between "+str(RULFrom)+" and "+str(RULTo)+" and BF_SD_TYPE='"+bfEventType+"' "
  rawdata = spark.sql(sql)
  data = rawdata.withColumn(colName,rawdata[colName].cast(DoubleType()))
  med = data.approxQuantile(colName, [0.5], 0.25)
  return med[0]


# COMMAND ----------

columnList = ['RESI_LL_LUBE_OIL_PRESS',
 'RESI_HH_LUBE_OIL_TEMP',
 'RESI_LL_SPEED',
 'RESI_HH_SPEED',
 'RESI_LL_VIBRATION',
 'RESI_HH_VIBRATION',
 'RESI_HH_THROW1_DIS_TEMP',
 'RESI_HH_THROW1_SUC_TEMP',
'LUBE_OIL_PRESS',
'LUBE_OIL_TEMP',
'THROW_1_DISC_PRESS',
'THROW_1_DISC_TEMP',
'THROW_1_SUC_PRESS',
'THROW_1_SUC_TEMP',
'THROW_2_DISC_PRESS',
'THROW_2_DISC_TEMP',
'THROW_2_SUC_PRESS',
'THROW_2_SUC_TEMP',
'THROW_3_DISC_PRESS',
'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS',
'THROW_3_SUC_TEMP',
'THROW_4_DISC_PRESS',
'THROW_4_DISC_TEMP',
'VIBRATION',
'CYL_1_TEMP',
'CYL_2_TEMP',
'CYL_3_TEMP',
'CYL_4_TEMP',
'CYL_5_TEMP',
'CYL_6_TEMP',
'CYL_7_TEMP',
'CYL_8_TEMP',
'CYL_9_TEMP',
'CYL_10_TEMP',
'CYL_11_TEMP',
'CYL_12_TEMP',
'LUBE_OIL_PRESS_ENGINE',
'MANI_PRESS',
'RIGHT_BANK_EXH_TEMP',
'SPEED',
'VIBRA_ENGINE',
'S1_RECY_VALVE',
'S1_SUCT_PRESS',
'S1_SUCT_TEMPE',
'S2_STAGE_DISC_PRESS',
'S2_SCRU_SCRUB_LEVEL',
'GAS_LIFT_HEAD_PRESS',
'IN_CONT_CONT_VALVE',
'IN_SEP_PRESS',
'WH_PRESSURE']

columnListDem = ['YEAR',
 'MONTH',
 'DAY',
 'HOUR',
 'MM',
 'CYCLE',
 'RUL',
 'EVENT_ID',
 'LABEL1',
 'LABEL2']

# COMMAND ----------

#6 - change column name
renamed_df = df.selectExpr(
"OBJECT_CODE as CODE",
"TIME as DAYTIME",
"N_YEAR as YEAR",
"N_MONTH as MONTH",
"N_DAY as DAY",
"N_HOUR as HOUR",
"N_MM as MM",
"CYCLE",
"ROUND(45 - LUBE_OIL_PRESS, 7) AS RESI_LL_LUBE_OIL_PRESS", 
"ROUND(LUBE_OIL_TEMP - 190,7) AS RESI_HH_LUBE_OIL_TEMP", 
"ROUND(820 - SPEED,7) AS RESI_LL_SPEED", 
"ROUND(SPEED - 1180,7) AS RESI_HH_SPEED", 
"ROUND(0.33 - VIBRA_ENGINE,7) AS RESI_LL_VIBRATION", 
"ROUND(VIBRA_ENGINE - 0.475,7) AS RESI_HH_VIBRATION", 
"ROUND(293.33 - THROW_1_DISC_TEMP,7) AS RESI_HH_THROW1_DIS_TEMP", 
"ROUND(THROW_1_SUC_TEMP - 190,7) AS RESI_HH_THROW1_SUC_TEMP",
'LUBE_OIL_PRESS',
'LUBE_OIL_TEMP',
'THROW_1_DISC_PRESS',
'THROW_1_DISC_TEMP',
'THROW_1_SUC_PRESS',
'THROW_1_SUC_TEMP',
'THROW_2_DISC_PRESS',
'THROW_2_DISC_TEMP',
'THROW_2_SUC_PRESS',
'THROW_2_SUC_TEMP',
'THROW_3_DISC_PRESS',
'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS',
'THROW_3_SUC_TEMP',
'THROW_4_DISC_PRESS',
'THROW_4_DISC_TEMP',
'VIBRATION',
'CYL_1_TEMP',
'CYL_2_TEMP',
'CYL_3_TEMP',
'CYL_4_TEMP',
'CYL_5_TEMP',
'CYL_6_TEMP',
'CYL_7_TEMP',
'CYL_8_TEMP',
'CYL_9_TEMP',
'CYL_10_TEMP',
'CYL_11_TEMP',
'CYL_12_TEMP',
#'FUEL_GAS_TEMP',
'LUBE_OIL_PRESS_ENGINE',
'MANI_PRESS',
'RIGHT_BANK_EXH_TEMP',
#'RUNNING_STATUS',
'SPEED',
'VIBRA_ENGINE',
'S1_RECY_VALVE',
'S1_SUCT_PRESS',
'S1_SUCT_TEMPE',
'S2_STAGE_DISC_PRESS',
'S2_SCRU_SCRUB_LEVEL',
'GAS_LIFT_HEAD_PRESS',
'IN_CONT_CONT_VALVE',
'IN_SEP_PRESS',
'WH_PRESSURE',
"RUL",
#"RUL*-1 as IN_RUL",
"EVENT_ID",
"LABEL1",
"LABEL2",
"BF_EVENT_TYPE as BF_SD_TYPE")

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data")
renamed_df = convertStr2Int(renamed_df,columnListDem)
renamed_df = replaceByMedian(renamed_df,columnList)
#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data")
## Filter out the data during RC S/D
renamed_df = spark.sql("select * from rc_data where EVENT_ID <> 9999999 and RUL <> 0 ")
renamed_df3 = spark.sql("select EVENT_ID, CODE , COUNT(*) as num_steps from rc_data where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")
renamed_df3.createOrReplaceTempView("tmp_step_event")
renamed_df3 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_data where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")

outlier_df = spark.sql("select EVENT_ID,CODE, COUNT(*) as num_error from rc_data where (LUBE_OIL_PRESS>52 and LUBE_OIL_PRESS <66 and LUBE_OIL_TEMP >= 138) and CODE not in ('BEWP-SK1060','BEWK-ZZZ-K0110A','MAWC-ME-C7400','BEWU-ME-U7400') and RUL between 60 and 1440 group by EVENT_ID, CODE")

outlier_df.createOrReplaceTempView("tmp_target_rc")

# COMMAND ----------

# MAGIC %md ## Data Exploration & Cleansing

# COMMAND ----------

# MAGIC %md First, let's look at the dataset dimension and data schema.

# COMMAND ----------

# check the dimensions of the data
df.count(), len(df.columns)

# COMMAND ----------

# # check whether the issue of df.show() is fixed
# display(df.show(1))

# COMMAND ----------

# check data schema
#df.dtypes
df.printSchema()

# COMMAND ----------

# display(df.describe())

# COMMAND ----------

# MAGIC %md #### Explanations on the data schema:
# MAGIC  * *** OBJECT_CODE***: Equipment Object Code
# MAGIC  * ***TIME***: Time Stamp of Each Data Point
# MAGIC  * ***N_YEAR***: Year of Time Stamp
# MAGIC  * ***N_MONTH***: Month of Time Stamp
# MAGIC  * ***N_DAY***:  Day of Time Stamp
# MAGIC  * ***N_HOUR***: Hours of Time Stamp
# MAGIC  * ***N_MM***: Minute of Time Stamp
# MAGIC  * ***LUBE_OIL_PRESS***: Lube Oil Pressure
# MAGIC  * ***LUBE_OIL_TEMP***: Lube Oil Temperature
# MAGIC  * ***THROW_1_DISC_PRESS***: Throw #1 Discharge Pressure
# MAGIC  * ***THROW_1_DISC_TEMP***: Throw #1 Discharge Temperature
# MAGIC  * ***THROW_1_SUC_PRESS***: Throw #1 Suction Pressure
# MAGIC  * ***THROW_1_SUC_TEMP***: Throw #1 Suction Temperature
# MAGIC  * ***THROW_2_DISC_PRESS***: Throw #2 Discharge Pressure
# MAGIC  * ***THROW_2_DISC_TEMP***: Throw #1 Discharge Temperature
# MAGIC  * ***THROW_2_SUC_PRESS***: Throw #1 Suction Pressure
# MAGIC  * ***THROW_2_SUC_TEMP***: Throw #2 Suction Pressure
# MAGIC  * ***THROW_3_DISC_PRESS***: Throw #3 Discharge Pressure
# MAGIC  * ***THROW_3_DISC_TEMP***: Throw #3 Discharge Temperature
# MAGIC  * ***THROW_3_SUC_PRESS***: Throw #3 Suction Pressure
# MAGIC  * ***THROW_3_SUC_TEMP***: Throw #3 Suction Temperature
# MAGIC  * ***THROW_4_DISC_PRESS***: Throw #4 Discharge Pressure
# MAGIC  * ***THROW_4_DISC_TEMP***: Throw #4 Discharge Temperature
# MAGIC  * ***VIBRATION***: Reciprocating Compressor Vibration
# MAGIC  * ***CYL_1_TEMP***: Cylinder #01 Temperature
# MAGIC  * ***CYL_2_TEMP***: Cylinder #02 Temperature
# MAGIC  * ***CYL_3_TEMP***: Cylinder #03 Temperature
# MAGIC  * ***CYL_4_TEMP***: Cylinder #04 Temperature
# MAGIC  * ***CYL_5_TEMP***: Cylinder #05 Temperature
# MAGIC  * ***CYL_6_TEMP***: Cylinder #06 Temperature
# MAGIC  * ***CYL_7_TEMP***: Cylinder #07 Temperature
# MAGIC  * ***CYL_8_TEMP***: Cylinder #08 Temperature
# MAGIC  * ***CYL_9_TEMP***: Cylinder #09 Temperature
# MAGIC  * ***CYL_10_TEMP***: Cylinder #10 Temperature
# MAGIC  * ***CYL_11_TEMP***: Cylinder #11 Temperature
# MAGIC  * ***CYL_12_TEMP***: Cylinder #12 Temperature
# MAGIC  * ***FUEL_GAS_TEMP***: Fuel Gas Temp
# MAGIC  * ***LUBE_OIL_PRESS_ENGINE***: Lube Oil Pressure - Engine
# MAGIC  * ***MANI_PRESS***: Manifold Pressure
# MAGIC  * ***RIGHT_BANK_EXH_TEMP***: Right Bank Exhaust Temperature
# MAGIC  * ***RUNNING_STATUS***: Running Status
# MAGIC  * ***SPEED***: Speed
# MAGIC  * ***VIBRA_ENGINE***: Reciprocating Engine Vibration - Engine
# MAGIC  * ***S1_RECY_VALVE***: 1st Stage Recycle Valve
# MAGIC  * ***S1_SUCT_PRESS***: 1st Stage Suction Pressure
# MAGIC  * ***S1_SUCT_TEMPE***: 1st Stage Suction Temperature
# MAGIC  * ***S2_STAGE_DISC_PRESS***: 2nd Stage Discharge Pressure
# MAGIC  * ***S2_SCRU_SCRUB_LEVEL***: 2nd Stage Suction Scrubber Level
# MAGIC  * ***GAS_LIFT_HEAD_PRESS***: Gas Lift Header Pressure
# MAGIC  * ***IN_CONT_CONT_VALVE***: Inlet Control Valve
# MAGIC  * ***IN_SEP_PRESS***: Inlet Separator Pressure
# MAGIC  * ***WH_PRESSURE***: Well Header Pressure
# MAGIC  * ***SD_TYPE***: Shutdown Type
# MAGIC  * ***CYCLE***: Cycle -- 1 is Start Running before
# MAGIC  * ***RUL***: Remaining Useful Life (Time to Failre)(Countdown timestep before failure)
# MAGIC  * ***LABEL1***: Lable 1 (Marked [0,1] is the tarket to predcit window time is 1440 time steps or 1 day) this table for binary classification
# MAGIC  * ***LABEL2***: Lable 2 Marked lables [0,1,2] for w0, w1 for multiclass classification
# MAGIC  * ***BF_EVENT_TYPE***: Category of Event Type Before Shutdown Event
# MAGIC  * ***EVENT_ID***: Event ID

# COMMAND ----------

# MAGIC %md #### As part of the data cleansing process, we standardized all the column names to lower case and replaced all the symbols with underscore. We also removed any duplicated records.

# COMMAND ----------

#--------------------------------------- initial data cleansing ---------------------------------------------#
# standardize the column names
def StandardizeNames(df):
    l = df.columns
    cols = [c.replace(' ','_').
              replace('[.]','_').
              replace('.','_').
              replace('[[:punct:]]','_').
              upper() for c in l]
    return df.toDF(*cols)
df = StandardizeNames(df)

# remove duplicated rows based on deviceid and date
# df = df.dropDuplicates(['deviceid', 'date'])

# #remove rows with missing deviceid, date
# df = df.dropna(how='any', subset=['deviceid', 'date'])

# df.select('deviceid','date').show(3)


# COMMAND ----------

#6 - change column name
renamed_df2 = renamed_df.selectExpr(
"CODE as CODE",
"DAYTIME",
"YEAR",
"MONTH",
"DAY",
"HOUR",
"MM",
"CYCLE",
'LUBE_OIL_PRESS AS S1',
'LUBE_OIL_TEMP  AS S2',
'THROW_1_DISC_PRESS  AS S3',
'THROW_1_DISC_TEMP  AS S4',
'THROW_1_SUC_PRESS AS S5',
'THROW_1_SUC_TEMP AS S6',
'THROW_2_DISC_PRESS AS S7',
'THROW_2_DISC_TEMP AS S8',
'THROW_2_SUC_PRESS AS S9',
'THROW_2_SUC_TEMP AS S10',
'THROW_3_DISC_PRESS AS S11',
'THROW_3_DISC_TEMP  AS S12',
'THROW_3_SUC_PRESS  AS S13',
'THROW_3_SUC_TEMP  AS S14',
'THROW_4_DISC_PRESS  AS S15',
'THROW_4_DISC_TEMP AS S16',
'VIBRATION  AS S17',
'CYL_1_TEMP  AS S18',
'CYL_2_TEMP AS S19',
'CYL_3_TEMP AS S21',
'CYL_4_TEMP AS S22',
'CYL_5_TEMP AS S23',
'CYL_6_TEMP AS S24',
'CYL_7_TEMP AS S25',
'CYL_8_TEMP AS S26',
'CYL_9_TEMP AS S27',
'CYL_10_TEMP AS S28',
'CYL_11_TEMP AS S29',
'CYL_12_TEMP AS S30',
#'FUEL_GAS_TEMP',
'LUBE_OIL_PRESS_ENGINE AS S31',
'MANI_PRESS AS S32',
'RIGHT_BANK_EXH_TEMP AS S33',
'SPEED AS S35',
'VIBRA_ENGINE AS S36',
'S1_RECY_VALVE AS S37',
'S1_SUCT_PRESS AS S38',
'S1_SUCT_TEMPE AS S39',
'S2_STAGE_DISC_PRESS AS S40',
'S2_SCRU_SCRUB_LEVEL AS S41',
'GAS_LIFT_HEAD_PRESS AS S42',
'IN_CONT_CONT_VALVE AS S43',
'IN_SEP_PRESS AS S44',
#'WH_PRESSURE AS S45',
"RUL as RUL",
"RUL*-1 as IN_RUL",
"EVENT_ID as EVENT_ID",
"LABEL1 as LABEL1",
"LABEL2 as LABEL2",
"BF_SD_TYPE")

# COMMAND ----------

# MAGIC %md #### Define groups of features -- date, categorical, numeric

# COMMAND ----------

#------------------------------------------- Define groups of features -----------------------------------------#

features_datetime = ['DAYTIME','YEAR','MONTH','DAY','HOUR','MM']
features_categorical = ['CODE','SD_TYPE','LABEL1', 'LABEL2', 'BF_EVENT_TYPE','BF_SD_TYPE']
features_timestep = ['YEAR','MONTH','DAY','HOUR','MM', 'CYCLE','RUL', 'IN_RUL']
features_numeric = list(set(renamed_df2.columns) -set(features_datetime)-set(features_categorical)-set(features_timestep))


# COMMAND ----------

features_numeric

# COMMAND ----------

# MAGIC %md #### Handling missing data

# COMMAND ----------

from pyspark.sql.functions import when

def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
        med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
        pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(med[0]))
        print("Relaced [",colName,"] by ", med[0])
    return pySparkDF
  
clean_df = replaceByMedian(renamed_df,features_numeric)

# COMMAND ----------

clean_df = clean_df.fillna(0, subset=features_numeric)
clean_df = clean_df.fillna("Unknown", subset=features_categorical)

# COMMAND ----------

# MAGIC %md #### For data exploration part, people usually would like to visualize the distribution of certain columns or the interation among columns. Here, we hand picked some columns to demonstrate how to do some basic visualizations.

# COMMAND ----------

#------------------------------------ data exploration and visualization ------------------------------------#

# Register dataframe as a temp table in SQL context
#12 - compute dataframe using sql command via string
clean_df.createOrReplaceTempView("rc_data")

renamed_df3 = spark.sql("select EVENT_ID, CODE , COUNT(*) as num_steps from rc_data where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")
renamed_df3.createOrReplaceTempView("tmp_step_event")
renamed_df3 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_data where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")

## Filter out the data during RC S/D
plotdata = spark.sql("select * from rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event WHERE num_steps > 420 ) ").toPandas()

# COMMAND ----------

display(renamed_df3)

# COMMAND ----------

clean_df = spark.sql("select * from rc_data ")

# COMMAND ----------

display(clean_df)

# COMMAND ----------

plotdata = spark.sql("select * from rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event WHERE num_steps > 420 ) AND YEAR=2018 AND MONTH=1 ")

# COMMAND ----------

display(plotdata)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC # show histogram distribution of some features
# MAGIC ax1 = plotdata[['S1']].plot(kind='hist', bins=22, facecolor='blue')
# MAGIC ax1.set_title('S1 distribution')
# MAGIC ax1.set_xlabel('S1'); ax1.set_ylabel('Counts');
# MAGIC plt.figure(figsize=(10,10)); plt.suptitle(''); 
# MAGIC #plt.show()
# MAGIC display()

# COMMAND ----------

ax1 = plotdata[['LUBE_OIL_PRESS_ENGINE']].plot(kind='hist', bins=21, facecolor='blue')
ax1.set_title('LUBE_OIL_PRESS_ENGINE distribution')
ax1.set_xlabel('LUBE_OIL_PRESS_ENGINE'); ax1.set_ylabel('Counts');
plt.figure(figsize=(10,10)); plt.suptitle(''); 
#plt.show()
display()

# COMMAND ----------

features = ['RESI_LL_LUBE_OIL_PRESS',
 'RESI_HH_LUBE_OIL_TEMP',
 'RESI_LL_SPEED',
 'RESI_HH_SPEED',
 'RESI_LL_VIBRATION',
 'RESI_HH_VIBRATION',
 'RESI_HH_THROW1_DIS_TEMP',
 'RESI_HH_THROW1_SUC_TEMP',
 'LUBE_OIL_PRESS',
 'LUBE_OIL_TEMP',
 'THROW_1_DISC_PRESS',
 'THROW_1_DISC_TEMP',
 'THROW_1_SUC_PRESS',
 'THROW_1_SUC_TEMP',
 'THROW_2_DISC_PRESS',
 'THROW_2_DISC_TEMP',
 'THROW_2_SUC_PRESS',
 'THROW_2_SUC_TEMP',
 'THROW_3_DISC_PRESS',
 'THROW_3_DISC_TEMP',
 'THROW_3_SUC_PRESS',
 'THROW_3_SUC_TEMP',
 'THROW_4_DISC_PRESS',
 'THROW_4_DISC_TEMP',
 'VIBRATION',
 'CYL_1_TEMP',
 'CYL_2_TEMP',
 'CYL_3_TEMP',
 'CYL_4_TEMP',
 'CYL_5_TEMP',
 'CYL_6_TEMP',
 'CYL_7_TEMP',
 'CYL_8_TEMP',
 'CYL_9_TEMP',
 'CYL_10_TEMP',
 'CYL_11_TEMP',
 'CYL_12_TEMP',
 'LUBE_OIL_PRESS_ENGINE',
 'MANI_PRESS',
 'RIGHT_BANK_EXH_TEMP',
 'RUNNING_STATUS',
 'SPEED',
 'VIBRA_ENGINE',
 'S1_RECY_VALVE',
 'S1_SUCT_PRESS',
 'S1_SUCT_TEMPE',
 'S2_STAGE_DISC_PRESS',
 'S2_SCRU_SCRUB_LEVEL',
 'GAS_LIFT_HEAD_PRESS',
 'IN_CONT_CONT_VALVE',
 'IN_SEP_PRESS',
 'WH_PRESSURE']

# COMMAND ----------

sensors = ['RUL',
  'IN_RUL',
 'S39',
 'S40',
 'S37',
 'S28',
 'S14',
 'S27',
 'S42',
 'S38',
 'S3',
 'S6',
 'S30',
 'S8',
 'S36',
 'S24',
 'S21',
 'S43',
 'S17',
 'S33',
 'S35',
 'S19',
 'S13',
 'S4',
 'S11',
 'S31',
 'S2',
 'S25',
 'S16',
 'EVENT_ID',
 'S32',
 'S5',
 'S9',
 'S29',
 'S18',
 'S41',
 'S26',
 'S1',
 'S22',
 'S7',
 'S23',
 'S44',
 'S12',
 'S10',
 'S15']

sensorsData = [
 'S39',
 'S40',
 'S37',
 'S28',
 'S14',
 'S27',
 'S42',
 'S38',
 'S3',
 'S6',
 'S30',
 'S8',
 'S36',
 'S24',
 'S21',
 'S43',
 'S17',
 'S33',
 'S35',
 'S19',
 'S13',
 'S4',
 'S11',
 'S31',
 'S2',
 'S25',
 'S16',
 'S32',
 'S5',
 'S9',
 'S29',
 'S18',
 'S41',
 'S26',
 'S1',
 'S22',
 'S7',
 'S23',
 'S44',
 'S12',
 'S10',
 'S15']

# COMMAND ----------

display(renamed_df2.select('EVENT_ID').where("EVENT_ID <> 0 ").distinct())

# COMMAND ----------

display(renamed_df2.select(sensors).where("EVENT_ID=4072 and RUL BETWEEN 1 and 1440"))

# COMMAND ----------

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)
## Filter out the data during RC S/D
plotdata = spark.sql("select * from rc_data where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event WHERE num_steps > 420 ) ").toPandas()
# show correlation matrix heatmap to explore some potential interesting patterns

# COMMAND ----------

data=renamed_df2.select(sensorsData).where("YEAR=2018 and MONTH=1").toPandas()

corr = data[sensorsData].corr()

# COMMAND ----------

#corr = np.corrcoef(plotdata[features])
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  ax = sns.heatmap(corr, mask=mask, vmax=0.3, square=True,center=1)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=1)
display()

# COMMAND ----------

def queryAVGSensorByInRUL(P_SD_TYPE, P_RUL_FROM, P_RUL_TO, P_MINIMUM_CYCLE):

    sparkDataFrame = spark.sql(" select IN_RUL, "
    +"AVG(RESI_LL_LUBE_OIL_PRESS) AS RESI_LL_LUBE_OIL_PRESS,"
    +"AVG( RESI_HH_LUBE_OIL_TEMP) AS  RESI_HH_LUBE_OIL_TEMP,"
    +"AVG( RESI_LL_SPEED) AS  RESI_LL_SPEED,"
    +"AVG( RESI_HH_SPEED) AS  RESI_HH_SPEED,"
    +"AVG( RESI_LL_VIBRATION) AS  RESI_LL_VIBRATION,"
    +"AVG( RESI_HH_VIBRATION) AS  RESI_HH_VIBRATION,"
    +"AVG( RESI_HH_THROW1_DIS_TEMP) AS  RESI_HH_THROW1_DIS_TEMP,"
    +"AVG( RESI_HH_THROW1_SUC_TEMP) AS  RESI_HH_THROW1_SUC_TEMP,"
    +"AVG( LUBE_OIL_PRESS) AS  LUBE_OIL_PRESS,"
    +"AVG( LUBE_OIL_TEMP) AS  LUBE_OIL_TEMP,"
    +"AVG( THROW_1_DISC_PRESS) AS  THROW_1_DISC_PRESS,"
    +"AVG( THROW_1_DISC_TEMP) AS  THROW_1_DISC_TEMP,"
    +"AVG( THROW_1_SUC_PRESS) AS  THROW_1_SUC_PRESS,"
    +"AVG( THROW_1_SUC_TEMP) AS  THROW_1_SUC_TEMP,"
    +"AVG( THROW_2_DISC_PRESS) AS  THROW_2_DISC_PRESS,"
    +"AVG( THROW_2_DISC_TEMP) AS  THROW_2_DISC_TEMP,"
    +"AVG( THROW_2_SUC_TEMP) AS  THROW_2_SUC_TEMP,"
    +"AVG( THROW_3_DISC_PRESS) AS  THROW_3_DISC_PRESS,"
    +"AVG( THROW_3_DISC_TEMP) AS  THROW_3_DISC_TEMP,"
    +"AVG( THROW_3_SUC_PRESS) AS  THROW_3_SUC_PRESS,"
    +"AVG( THROW_3_SUC_TEMP) AS  THROW_3_SUC_TEMP,"
    +"AVG( THROW_4_DISC_PRESS) AS  THROW_4_DISC_PRESS,"
    +"AVG( THROW_4_DISC_TEMP) AS  THROW_4_DISC_TEMP,"
    +"AVG( VIBRATION) AS  VIBRATION,"
    +"AVG( CYL_1_TEMP) AS  CYL_1_TEMP,"
    +"AVG( CYL_2_TEMP) AS  CYL_2_TEMP,"
    +"AVG( CYL_3_TEMP) AS  CYL_3_TEMP,"
    +"AVG( CYL_4_TEMP) AS  CYL_4_TEMP,"
    +"AVG( CYL_5_TEMP) AS  CYL_5_TEMP,"
    +"AVG( CYL_6_TEMP) AS  CYL_6_TEMP,"
    +"AVG( CYL_7_TEMP) AS  CYL_7_TEMP,"
    +"AVG( CYL_8_TEMP) AS  CYL_8_TEMP,"
    +"AVG( CYL_9_TEMP) AS  CYL_9_TEMP,"
    +"AVG( CYL_10_TEMP) AS  CYL_10_TEMP,"
    +"AVG( CYL_11_TEMP) AS  CYL_11_TEMP,"
   ## +"AVG( FUEL_GAS_PRESS) AS  FUEL_GAS_PRESS,"
    +"AVG( LUBE_OIL_PRESS_ENGINE) AS  LUBE_OIL_PRESS_ENGINE,"
    +"AVG( SPEED) AS  SPEED,"
    +"AVG( VIBRA_ENGINE) AS  VIBRA_ENGINE,"
    +"AVG(S1_RECY_VALVE) AS S1_RECY_VALVE,"
    +"AVG(S1_SUCT_PRESS) AS S1_SUCT_PRESS,"
    +"AVG(S1_SUCT_TEMPE) AS S1_SUCT_TEMPE,"
    +"AVG(S2_STAGE_DISC_PRESS) AS S2_STAGE_DISC_PRESS,"
    +"AVG(S2_SCRU_SCRUB_LEVEL) AS S2_SCRU_SCRUB_LEVEL,"
    +"AVG(GAS_LIFT_HEAD_PRESS) AS GAS_LIFT_HEAD_PRESS,"
    +"AVG(IN_CONT_CONT_VALVE) AS IN_CONT_CONT_VALVE,"
    +"AVG(IN_SEP_PRESS) AS IN_SEP_PRESS,"
    +"AVG(WH_PRESSURE) AS WH_PRESSURE"
    + " FROM rc_data where YEAR between 2016 and 2018 "
    + " AND RUL between "+ str(P_RUL_FROM) +" AND "+ str(P_RUL_TO)  +" AND BF_SD_TYPE = '"+ P_SD_TYPE +"' " 
    + " AND EVENT_ID <> 9999999 and CYCLE <> 9999999 and RUL<> 0 "
    + " AND EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event WHERE num_steps > "+ str(P_MINIMUM_CYCLE)+") "
    + " GROUP BY IN_RUL order by IN_RUL"
                         )
    return sparkDataFrame

# COMMAND ----------

# pick a large window size of 50 cycles
sampleEvent = renamed_df2.select(sensors).where("EVENT_ID=4072 and RUL BETWEEN 1 and 1440")
sampleEvent = sampleEvent.toPandas()
sampleEvent = sampleEvent.sort_values(['RUL'])

# COMMAND ----------

colNameSet1 =['RUL','S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
engineSample50Set1 = renamed_df2.select(colNameSet1).where("EVENT_ID=4072 and RUL BETWEEN 1 and 1440").toPandas()
engineSample50Set1 = engineSample50Set1.sort_values(['RUL'])
engineSample50Set1 = engineSample50Set1.sort_values(['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'])

# COMMAND ----------

colNameSet2 = ['THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS','THROW_3_DISC_TEMP', 'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION']
colNameSet3 = ['CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP', 'CYL_6_TEMP']
colNameSet4 = ['CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP', 'LUBE_OIL_PRESS_ENGINE', 'SPEED']
# preparing data for visualizations 
# window of 50 cycles prior to a failure point for engine id 3
#engineSample = test_df[test_df['EVENT_ID'] == 3]
engineSample50cycleWindow = sampleEvent[sampleEvent['RUL'] <= sampleEvent['RUL'].min() + sequence_length]
engineSample50Set2 = engineSample50cycleWindow[colNameSet2]
engineSample50Set3 = engineSample50cycleWindow[colNameSet3]
engineSample50Set4 = engineSample50cycleWindow[colNameSet4]

# COMMAND ----------

# plotting sensor data for engine ID 3 prior to a failure point - sensors 1-10 
#newDF = engineSample50Set1['LUBE_OIL_PRESS']
ax1 = engineSample50Set1.plot(subplots=True, sharex=True, figsize=(20,20))
#ax1.set_title('Sensor Value from Last 1,440 Time Step Before Failure')
#ax1.set_xlabel('Cycle 1 Minute Per Step')
#ax1.set_ylabel('Sensor Value')
display()

# COMMAND ----------

engineSample50Set1.plot.scatter(x='THROW_1_SUC_TEMP', y='THROW_1_SUC_PRESS', color='DarkGreen', label='Group 2', ax=ax); 
display()

# COMMAND ----------

ax = engineSample50Set2.plot.scatter(x='THROW_3_SUC_TEMP', y='THROW_3_SUC_PRESS', color='DarkBlue', label='Group 1');
engineSample50Set1.plot.scatter(x='THROW_1_SUC_TEMP', y='THROW_1_SUC_PRESS', color='DarkGreen', label='Group 2', ax=ax); 
  
display()

# COMMAND ----------

# plotting sensor data for engine ID 3 prior to a failure point - sensors 11-21 
ax2 = engineSample50Set2.plot(subplots=True, sharex=True, figsize=(20,20))
display()

# COMMAND ----------

# plotting sensor data for engine ID 3 prior to a failure point - sensors 11-21 
ax3 = engineSample50Set3.plot(subplots=True, sharex=True, figsize=(20,20))
display()

# COMMAND ----------

# plotting sensor data for engine ID 3 prior to a failure point - sensors 11-21 
ax4 = engineSample50Set4.plot(subplots=True, sharex=True, figsize=(20,20))
display()

# COMMAND ----------

import seaborn as sns
sns.set(style="white")

#df = sns.load_dataset("iris")
g = sns.PairGrid(engineSample50Set1, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)

g.map_upper(sns.regplot)

display(g.fig)

# COMMAND ----------

g = sns.PairGrid(engineSample50Set2, diag_sharey=False)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)

g.map_upper(sns.regplot)

display(g.fig)

# COMMAND ----------

def convertPasdasToSeries(df):
  result = ""
  if df.empty:
      # Empty dataframe, so convert to empty Series.
      result = pd.Series()
  elif df.shape == (1, 1):
      # DataFrame with one value, so convert to series with appropriate index.
      result = pd.Series(df.iat[0, 0], index=df.columns)
  elif len(df) == 1:
      # Convert to series per OP's question.
      result = df.T.squeeze()
  else:
      # Dataframe with multiple rows.  Implement desired behavior.
      result = pd.Series() if df.empty else df.iloc[0, :]
  return result

# COMMAND ----------

from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot

series = convertPasdasToSeries(sampleEvent[columns])
#Series.from_csv('daily-minimum-temperatures.csv', header=0)
print(series)

# COMMAND ----------

# pick a large window size of 50 cycles
sampleTimeSeries = spark.sql("select EVENT_ID, CYCLE, IN_RUL, LUBE_OIL_PRESS,THROW_1_SUC_PRESS from rc_data where CODE = 'BEWF-ZZZ-F0110A' AND LABEL1=1 ORDER BY EVENT_ID, CYCLE, IN_RUL ")

# COMMAND ----------

display(sampleTimeSeries)

# COMMAND ----------

P_SD_TYPES = ["NMD","SCM","STB","SPM","UCM"]
P_START_RUL = 1
P_END_RUL = 1440
P_MINIMUM_RUL = 1440
P_CUR_SD_TYPE = "NMD"

renamed_df = queryAVGSensorByInRUL(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) 
pd_df_1 = (renamed_df.toPandas())

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir RC
# MAGIC mkdir RC/img

# COMMAND ----------

# MAGIC %sh
# MAGIC pwd

# COMMAND ----------

path = "/databricks/driver/RC/"

# COMMAND ----------

import pandas as pd

xLabel = "Inverse Remaining Useful Life (Minute)"

def plot_rolling_average(xSortedList,ySortedList, w=15,p_sd_type="UCM"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    '''
    Plot rolling mean and rolling standard deviation for a given time series and window
    '''
    # calculate moving averages
    rolling_mean = ySortedList.rolling(window=w).mean()  #pd.rolling_mean(y, window=w)
    #rolling_mean = pd.rolling_mean(y, window=w)
    rolling_std = ySortedList.rolling(window=w).std()
    #rolling_mean = ts_log.rolling(w).mean()
    
    # plot statistics
    plt.plot(xSortedList, ySortedList , label='Original')
    plt.plot(xSortedList,rolling_mean, color='crimson', label='Moving average mean')
    plt.plot(xSortedList,rolling_std, color='darkslateblue', label='Moving average SD')
    plt.legend(loc='best')
    plt.title(p_sd_type+' Rolling Mean & SD of '+ ySortedList.name, fontsize=22)
    plt.xlabel(xLabel)
    plt.ylabel("Sensor Value")
    plt.savefig(path+"img/"+timestamp+"_"+p_sd_type+"_"+ySortedList.name+str(w)+"_moving_average.png")
    #plt.show(block=False)
    #display(plt)
    return

# COMMAND ----------

def plot_rolling_average_4wSensors(x,y1,y2,y3,y4, w=15,p_sd_type="UCM"):
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # define figure and axes
    fig, axes = plt.subplots(2, 2, sharey=False, sharex=False);
    fig.set_figwidth(14);
    fig.set_figheight(8);
    
    # push data to each ax
    #upper left
    axes[0][0].plot(x, y1, label='Original');
    axes[0][0].plot(x, y1.rolling(w).mean(), label='Rolling Mean & SD of '+ y1.name , color='crimson');
    axes[0][0].plot(x, y1.rolling(w).std(), label='Moving average standard deviation'+str(w)+' window time (m)', color='darkslateblue');


    axes[0][0].set_xlabel(xLabel);
    axes[0][0].set_ylabel("Sensor Value");
    axes[0][0].set_title('Rolling Mean & SD of '+ y1.name,fontsize=18);
    axes[0][0].legend(loc='best');

    # upper right
    axes[0][1].plot(x, y2, label='Original')
    axes[0][1].plot(x, y2.rolling(w).mean(), label='Rolling Mean & SD of '+ y2.name +str(w)+' window time (m)', color='crimson');
    axes[0][1].plot(x, y2.rolling(w).std(), label='Moving average standard deviation', color='darkslateblue');

    axes[0][1].set_xlabel(xLabel);
    axes[0][1].set_ylabel("Sensor Value");
    axes[0][1].set_title('Rolling Mean & SD of '+ y2.name,fontsize=18);
    axes[0][1].legend(loc='best');

    # lower left
    axes[1][0].plot(x, y3, label='Original');
    axes[1][0].plot(x, y3.rolling(w).mean(), label='Rolling Mean & SD of '+ y3.name +str(w)+' window time (m)', color='crimson');
    axes[1][0].plot(x, y3.rolling(w).std(), label='Moving average standard deviation', color='darkslateblue');
    axes[1][0].set_xlabel(xLabel);
    axes[1][0].set_ylabel("Sensor Value");
    axes[1][0].set_title('Rolling Mean & SD of '+ y3.name,fontsize=18);
    axes[1][0].legend(loc='best');

    # lower right
    axes[1][1].plot(x, y4, label='Original');
    axes[1][1].plot(x, y4.rolling(w).mean(), label='Rolling Mean & SD of '+ y4.name +str(w)+' window time (m)', color='crimson');
    axes[1][1].plot(x, y4.rolling(w).std(), label='Moving average standard deviation', color='darkslateblue');
    axes[1][1].set_xlabel(xLabel);
    axes[1][1].set_ylabel("Sensor Value");
    axes[1][1].set_title('Rolling Mean & SD of '+ y4.name,fontsize=18);
    axes[1][1].legend(loc='best');
    plt.tight_layout();
    plt.savefig(path+"img/"+timestamp+"_"+p_sd_type+"_"+"rolling_windows.png")
    #plt.show()

    return

# COMMAND ----------


def plot_rolling_average_4w(x,y, w=15,p_sd_type="UCM"):
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # define figure and axes
    fig, axes = plt.subplots(2, 2, sharey=False, sharex=False);
    fig.set_figwidth(14);
    fig.set_figheight(8);

    # push data to each ax
    #upper left
    axes[0][0].plot(y.index, y, label='Original');
    axes[0][0].plot(y.index, y.rolling(window=15).mean(), label='15-Cycle/RUL Rolling Mean', color='crimson');
    axes[0][0].plot(y.index, y.rolling(window=15).std(), label='Moving average standard deviation', color='darkslateblue');


    axes[0][0].set_xlabel("Cycle");
    axes[0][0].set_ylabel("Sensor Value");
    axes[0][0].set_title("15-Cycle/RUL Moving Average");
    axes[0][0].legend(loc='best');

    # upper right
    axes[0][1].plot(y.index, y, label='Original')
    axes[0][1].plot(y.index, y.rolling(window=30).mean(), label='30-Cycle/RUL Rolling Mean', color='crimson');
    axes[0][1].plot(y.index, y.rolling(window=30).std(), label='Moving average standard deviation', color='darkslateblue');

    axes[0][1].set_xlabel("Cycle");windowtime = 15

def drawGraphBySDType(cur_sd_type):
    
    ## Moviong Avg 15 time stpes....
    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],
                                   pd_df_1['RESI_LL_LUBE_OIL_PRESS'],
                                   pd_df_1['RESI_HH_LUBE_OIL_TEMP'],
                                   pd_df_1['RESI_LL_SPEED'],
                                   pd_df_1['RESI_HH_SPEED'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],
                                   pd_df_1['RESI_LL_VIBRATION'],
                                   pd_df_1['RESI_HH_VIBRATION'],
                                   pd_df_1['RESI_HH_THROW1_DIS_TEMP'],
                                   pd_df_1['RESI_HH_THROW1_SUC_TEMP'],windowtime,cur_sd_type)


    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['LUBE_OIL_PRESS'],
                                   pd_df_1['LUBE_OIL_TEMP'],
                                   pd_df_1['THROW_1_DISC_PRESS'],pd_df_1['THROW_1_DISC_TEMP'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['THROW_1_SUC_PRESS'],
                                   pd_df_1['THROW_1_SUC_TEMP'],
                                   pd_df_1['THROW_2_DISC_PRESS'],pd_df_1['THROW_2_DISC_TEMP'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['THROW_2_SUC_TEMP'],
                                   pd_df_1['THROW_3_DISC_PRESS'],
                                   pd_df_1['THROW_3_DISC_TEMP'],pd_df_1['THROW_3_SUC_PRESS'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['THROW_3_SUC_TEMP'],
                                   pd_df_1['THROW_4_DISC_PRESS'],
                                   pd_df_1['THROW_4_DISC_TEMP'],pd_df_1['VIBRATION'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['CYL_1_TEMP'],
                                   pd_df_1['CYL_2_TEMP'],
                                   pd_df_1['CYL_3_TEMP'],pd_df_1['CYL_4_TEMP'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['CYL_5_TEMP'],
                                   pd_df_1['CYL_6_TEMP'],
                                   pd_df_1['CYL_7_TEMP'],pd_df_1['CYL_8_TEMP'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['CYL_9_TEMP'],
                                   pd_df_1['CYL_10_TEMP'],
                                   pd_df_1['CYL_11_TEMP'],pd_df_1['FUEL_GAS_PRESS'],windowtime,cur_sd_type)

    plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['LUBE_OIL_PRESS_ENGINE'],
                                   pd_df_1['SPEED'],
                                   pd_df_1['VIBRA_ENGINE'],pd_df_1['VIBRA_ENGINE'],windowtime,cur_sd_type)
    return
    axes[0][1].set_ylabel("Sensor Value");
    axes[0][1].set_title("30-Cycle/RUL Moving Average",fontsize=18);
    axes[0][1].legend(loc='best');

    # lower left
    axes[1][0].plot(y.index, y, label='Original');
    axes[1][0].plot(y.index, y.rolling(window=45).mean(), label='45-Cycle/RUL Rolling Mean', color='crimson');
    axes[1][0].plot(y.index, y.rolling(window=45).std(), label='Moving average standard deviation', color='darkslateblue');
    axes[1][0].set_xlabel("Cycle");
    axes[1][0].set_ylabel("Sensor Value");
    axes[1][0].set_title("45-Cycle/RUL Moving Average",fontsize=18);
    axes[1][0].legend(loc='best');

    # lower right
    axes[1][1].plot(y.index, y, label='Original');
    axes[1][1].plot(y.index, y.rolling(window=60).mean(), label='60-Cycle/RUL Rolling Mean', color='crimson');
    axes[1][1].plot(y.index, y.rolling(window=60).std(), label='Moving average standard deviation', color='darkslateblue');
    axes[1][1].set_xlabel("Cycle");
    axes[1][1].set_ylabel("Sensor Value");
    axes[1][1].set_title("60-Cycle/RUL Moving Average",fontsize=18);
    axes[1][1].legend(loc='best');
    plt.tight_layout();
    plt.savefig(path+'img/'+timestamp+'_'+p_sd_type+'_'+'rolling_windows.png')
    ##plt.show()

    return

# COMMAND ----------

P_SD_TYPES = ["UCM"]

P_START_RUL = 1
P_END_RUL = 1440
P_MINIMUM_RUL = 420
P_CUR_SD_TYPE = "UCM"
windowtime = 15
selectedCols = features

for P_CUR_SD_TYPE in P_SD_TYPES: 
    renamed_df = queryAVGSensorByInRUL(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) 
    pd_df_1 = (renamed_df.toPandas())
    #drawGraphBySDType(P_CUR_SD_TYPE)
    for colName in selectedCols:
        #pd_df_1[colName].head()
        plot_rolling_average(pd_df_1['IN_RUL'], pd_df_1[colName],windowtime,P_CUR_SD_TYPE)

print("**Finished**")

# COMMAND ----------

# MAGIC %sh
# MAGIC #ls -al RCPredictiveMA/img
# MAGIC zip RCPredictiveMA_MA3.zip RCPredictiveMA/img/*.png

# COMMAND ----------

def queryAVGSensorByEventId(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) :
    df_simple2 = spark.sql(" select  IN_RUL ,MONTH, EVENT_ID ,"
                            +"AVG( RESI_LL_LUBE_OIL_PRESS) AS RESI_LL_LUBE_OIL_PRESS,"
                            +"AVG( RESI_HH_LUBE_OIL_TEMP) AS  RESI_HH_LUBE_OIL_TEMP,"
                            +"AVG( RESI_LL_SPEED) AS  RESI_LL_SPEED,"
                            +"AVG( RESI_HH_SPEED) AS  RESI_HH_SPEED,"
                            +"AVG( RESI_LL_VIBRATION) AS  RESI_LL_VIBRATION,"
                            +"AVG( RESI_HH_VIBRATION) AS  RESI_HH_VIBRATION,"
                            +"AVG( RESI_HH_THROW1_DIS_TEMP) AS  RESI_HH_THROW1_DIS_TEMP,"
                            +"AVG( RESI_HH_THROW1_SUC_TEMP) AS  RESI_HH_THROW1_SUC_TEMP,"
                            +"AVG( LUBE_OIL_PRESS) AS  LUBE_OIL_PRESS,"
                            +"AVG( LUBE_OIL_TEMP) AS  LUBE_OIL_TEMP,"
                            +"AVG( THROW_1_DISC_PRESS) AS  THROW_1_DISC_PRESS,"
                            +"AVG( THROW_1_DISC_TEMP) AS  THROW_1_DISC_TEMP,"
                            +"AVG( THROW_1_SUC_PRESS) AS  THROW_1_SUC_PRESS,"
                            +"AVG( THROW_1_SUC_TEMP) AS  THROW_1_SUC_TEMP,"
                            +"AVG( THROW_2_DISC_PRESS) AS  THROW_2_DISC_PRESS,"
                            +"AVG( THROW_2_DISC_TEMP) AS  THROW_2_DISC_TEMP,"
                            +"AVG( THROW_2_SUC_TEMP) AS  THROW_2_SUC_TEMP,"
                            +"AVG( THROW_3_DISC_PRESS) AS  THROW_3_DISC_PRESS,"
                            +"AVG( THROW_3_DISC_TEMP) AS  THROW_3_DISC_TEMP,"
                            +"AVG( THROW_3_SUC_PRESS) AS  THROW_3_SUC_PRESS,"
                            +"AVG( THROW_3_SUC_TEMP) AS  THROW_3_SUC_TEMP,"
                            +"AVG( THROW_4_DISC_PRESS) AS  THROW_4_DISC_PRESS,"
                            +"AVG( THROW_4_DISC_TEMP) AS  THROW_4_DISC_TEMP,"
                            +"AVG( VIBRATION) AS  VIBRATION,"
                            +"AVG( CYL_1_TEMP) AS  CYL_1_TEMP,"
                            +"AVG( CYL_2_TEMP) AS  CYL_2_TEMP,"
                            +"AVG( CYL_3_TEMP) AS  CYL_3_TEMP,"
                            +"AVG( CYL_4_TEMP) AS  CYL_4_TEMP,"
                            +"AVG( CYL_5_TEMP) AS  CYL_5_TEMP,"
                            +"AVG( CYL_6_TEMP) AS  CYL_6_TEMP,"
                            +"AVG( CYL_7_TEMP) AS  CYL_7_TEMP,"
                            +"AVG( CYL_8_TEMP) AS  CYL_8_TEMP,"
                            +"AVG( CYL_9_TEMP) AS  CYL_9_TEMP,"
                            +"AVG( CYL_10_TEMP) AS  CYL_10_TEMP,"
                            +"AVG( CYL_11_TEMP) AS  CYL_11_TEMP,"
                           ## +"AVG( FUEL_GAS_PRESS) AS  FUEL_GAS_PRESS,"
                            +"AVG( LUBE_OIL_PRESS_ENGINE) AS  LUBE_OIL_PRESS_ENGINE,"
                            +"AVG( SPEED) AS  SPEED,"
                            +"AVG( VIBRA_ENGINE) AS  VIBRA_ENGINE "
                          + " from rc_data where "
                          + " RUL between "+ str(P_START_RUL) +" and "+str(P_END_RUL) 
                          + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 and RUL<>0 "
                          + " and BF_SD_TYPE ='" + P_CUR_SD_TYPE+ "'"
                          + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > "+str(P_MINIMUM_RUL)+" )"
                          + " group by IN_RUL ,MONTH, EVENT_ID "
                          + " order by IN_RUL ,MONTH, EVENT_ID ")
    return df_simple2


# COMMAND ----------

# # create nice axes names
# month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')
# # reshape date

def plot_boxPlot(pivot_df, parameterName,sd_type,rul_start,rul_end):
    ##print(df_piv_box)
    # create a box plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    imgPathBoxPlot = path+'img/'+sd_type+'_SEL_BOX_'+timestamp+"_"+parameterName+"_"+str(rul_start)+"_"+str(rul_end)+'.png'
    imgPathHeat = path+'img/'+sd_type+'_SEL_HEATMAP_'+timestamp+"_"+parameterName+"_"+str(rul_start)+"_"+str(rul_end)+'.png'
    
#     fig, ax = plt.subplots()
#     pivot_df.plot(ax=ax, kind='box')
#     ax.set_title('Seasonal Effect per Month', fontsize=24)
#     ax.set_xlabel('Month')
#     ax.set_ylabel(parameterName)
#     ax.xaxis.set_ticks_position('bottom')
#     fig.tight_layout()
#     plt.savefig(imgPathBoxPlot)
#     plt.show()
    # plot heatmap
    
    sns_heatmap = sns.heatmap(pivot_df, annot=False)
    #display(fig)
    fig = sns_heatmap.get_figure()
    fig.savefig(imgPathHeat)
    return

# COMMAND ----------

# create line plot
def plot_line_graph(pivot_df, parameterName,sd_type,rul_start,rul_end):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    imgPath = path+'img/'+sd_type+'_SEL_LINE_'+timestamp+"_"+parameterName+"_"+str(rul_start)+"_"+str(rul_end)+'.png'
    
    pivot_df.plot(colormap='jet');
    plt.title("Line Plot of Sensor: "+parameterName+"("+sd_type+")"+str(rul_start)+" to "+str(rul_end), fontsize=18)
    plt.ylabel('Sensor Value')
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(imgPath)
    ##plt.show()
    #display(plt)
    return

# COMMAND ----------

selectedCols3 = ['RESI_LL_LUBE_OIL_PRESS',
 'RESI_HH_LUBE_OIL_TEMP',
 'RESI_LL_SPEED',
 'RESI_HH_SPEED',
 'RESI_LL_VIBRATION',
 'RESI_HH_VIBRATION',
 'RESI_HH_THROW1_DIS_TEMP',
 'RESI_HH_THROW1_SUC_TEMP',
 'LUBE_OIL_PRESS',
 'LUBE_OIL_TEMP',
 'THROW_1_DISC_PRESS',
 'THROW_1_DISC_TEMP',
 'THROW_1_SUC_PRESS',
 'THROW_1_SUC_TEMP',
 'THROW_2_DISC_PRESS',
 'THROW_2_DISC_TEMP',
 'THROW_2_SUC_TEMP',
 'THROW_3_DISC_PRESS',
 'THROW_3_DISC_TEMP',
 'THROW_3_SUC_PRESS',
 'THROW_3_SUC_TEMP',
 'THROW_4_DISC_PRESS',
 'THROW_4_DISC_TEMP',
 'VIBRATION',
 'CYL_1_TEMP',
 'CYL_2_TEMP',
 'CYL_3_TEMP',
 'CYL_4_TEMP',
 'CYL_5_TEMP',
 'CYL_6_TEMP',
 'CYL_7_TEMP',
 'CYL_8_TEMP',
 'CYL_9_TEMP',
 'CYL_10_TEMP',
 'CYL_11_TEMP',
 ##'FUEL_GAS_PRESS',
 'LUBE_OIL_PRESS_ENGINE',
 'SPEED',
 'VIBRA_ENGINE']

#P_SD_TYPES = ["NMD","SCM","STB","SPM","UCM"]
P_SD_TYPES = ["UCM"]
P_START_RUL = 1
P_END_RUL = 1440
P_MINIMUM_RUL = 720
P_CUR_SD_TYPE = "UCM"
selectedCols = selectedCols3 

for P_CUR_SD_TYPE in P_SD_TYPES: 
    renamed_df = queryAVGSensorByEventId(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL)
    print("** Start drawing graph of "+P_CUR_SD_TYPE)
    
    pd_df_1 = (renamed_df.toPandas())
    for colName in selectedCols3:
        df_piv_box = pd_df_1.pivot_table(index=['IN_RUL'], columns='EVENT_ID',values= colName , aggfunc='mean')
        plot_line_graph(df_piv_box,colName,P_CUR_SD_TYPE,P_START_RUL,P_END_RUL)
        print("** Saved line graph of "+colName)
        df_piv_box = pd_df_1.pivot_table(index=['MONTH'], columns='IN_RUL',values= colName , aggfunc='mean')
        plot_boxPlot(df_piv_box,colName,P_CUR_SD_TYPE,P_START_RUL,P_END_RUL)
        print("** Saved box and heatmap graphs of "+colName)
        
    print("** End drawing graph of "+P_CUR_SD_TYPE)

print("**Finish all drawing graph**")


# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -al /databricks/driver/RC/img

# COMMAND ----------

# MAGIC %sh
# MAGIC zip RC_LINE_MAIN_DATA_01.zip /databricks/driver/RC/img/*png

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/RC_LINE_MAIN_DATA_01.zip", "dbfs:/mnt/Exploratory/WCLD/02_MAIN_DATA_01/DataAnalysis")

# COMMAND ----------

# MAGIC %fs
# MAGIC mkdirs dbfs:/FileStore/images/RC_DATA_ANALYSIS/RC_LINE_MAIN_DATA_01

# COMMAND ----------

# MAGIC %fs
# MAGIC cp -r file:/databricks/driver/RC/img/ dbfs:/FileStore/images/RC_DATA_ANALYSIS/RC_LINE_MAIN_DATA_01

# COMMAND ----------

sample_img_dir = "dbfs:/FileStore/images/RC_DATA_ANALYSIS/RC_LINE_MAIN_DATA_01"
listImg = dbutils.fs.ls(sample_img_dir)

# COMMAND ----------

display(listImg)

# COMMAND ----------

# %fs put -f "file:/databricks/driver/RCPredictiveMA/img/" "dbfs:/RC/exploratory"
%sh
# # List files in DBFS
# dbfs ls
# # Put local file ./apple.txt to dbfs:/apple.txt
# dbfs cp ./apple.txt dbfs:/apple.txt
# # Get dbfs:/apple.txt and save to local file ./apple.txt
# dbfs cp dbfs:/apple.txt ./apple.txt
# # Recursively put local dir ./banana to dbfs:/banana
dbfs cp -r ./databricks/driver/RCPredictiveMA/img/ dbfs:/RC/exploratory

# COMMAND ----------

def displayImgDBFS(tfb_log_dir,dbfsImgPath, containsKey1, containsKey2,imgSizeX=1280,imgSizeY=640):
  dataFrameLogs = dbutils.fs.ls(tfb_log_dir)
  line = []
  htmlStr = "<table style='width:100%'>"
  
  # paths = dataFrameLogs[0]
  for path in dataFrameLogs:
    line = path

    if line[1].__contains__(containsKey1) and line[1].__contains__(containsKey2):
      filePath = dbfsImgPath + line[1]
      
      print (filePath)
      htmlStr = htmlStr + "<tr><img src='"+filePath+"' style='width:"+str(imgSizeX)+"px;height:"+str(imgSizeY)+"px;'></tr>"
      htmlStr = htmlStr+"<tr>"+filePath+"</tr>"

  htmlStr=htmlStr+"</tr></table>"
  return htmlStr

# COMMAND ----------


tfb_log_dir = "dbfs:/FileStore/images/RC_DATA_ANALYSIS/RC_LINE_MAIN_DATA_01"
dbfsImgPath = "files/images/RC_DATA_ANALYSIS/RC_LINE_MAIN_DATA_01/"
containsKey1 ="UCM_SEL_LINE"
containsKey2 ="LUBE_OIL_PRESS_"
htmpString = displayImgDBFS(tfb_log_dir,dbfsImgPath, containsKey1, containsKey2,1280,640)

# COMMAND ----------

displayHTML(htmpString)

# COMMAND ----------

#/FileStore/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM/11_M9_LSTM_model_verify.png

tfb_log_dir = "dbfs:/FileStore/images/RC_DATA_ANALYSIS/RC_LINE_MAIN_DATA_01/"
dbfsImgPath = "files/images/RC_DATA_ANALYSIS/RC_LINE_MAIN_DATA_01/"
containsKey1 ="_PRESS"
containsKey2 ="_1_1440"
htmpString = displayImgDBFS(tfb_log_dir,dbfsImgPath, containsKey1, containsKey2,840,420)

# COMMAND ----------

displayHTML(htmpString)

# COMMAND ----------

colSet1 = ['RESI_LL_LUBE_OIL_PRESS',
 'RESI_HH_LUBE_OIL_TEMP',
 'RESI_LL_SPEED',
 'RESI_HH_SPEED']

colSet2 = [
 'RESI_LL_VIBRATION',
 'RESI_HH_VIBRATION',
 'RESI_HH_THROW1_DIS_TEMP',
 'RESI_HH_THROW1_SUC_TEMP']

colSet3= [
 'LUBE_OIL_PRESS',
 'LUBE_OIL_TEMP',
 'THROW_1_DISC_PRESS',
 'THROW_1_DISC_TEMP']

colSet4 = [
 'THROW_1_SUC_PRESS',
 'THROW_1_SUC_TEMP',
 'THROW_2_DISC_PRESS',
 'THROW_2_DISC_TEMP',
 'THROW_2_SUC_PRESS',
 'THROW_2_SUC_TEMP',
 'THROW_3_DISC_PRESS',
 'THROW_3_DISC_TEMP',
 'THROW_3_SUC_PRESS',
 'THROW_3_SUC_TEMP',
 'THROW_4_DISC_PRESS',
 'THROW_4_DISC_TEMP',
 'VIBRATION',
 'CYL_1_TEMP',
 'CYL_2_TEMP',
 'CYL_3_TEMP',
 'CYL_4_TEMP',
 'CYL_5_TEMP',
 'CYL_6_TEMP',
 'CYL_7_TEMP',
 'CYL_8_TEMP',
 'CYL_9_TEMP',
 'CYL_10_TEMP',
 'CYL_11_TEMP',
 'CYL_12_TEMP',
 'LUBE_OIL_PRESS_ENGINE',
 'MANI_PRESS',
 'RIGHT_BANK_EXH_TEMP',
 'RUNNING_STATUS',
 'SPEED',
 'VIBRA_ENGINE',
 'S1_RECY_VALVE',
 'S1_SUCT_PRESS',
 'S1_SUCT_TEMPE',
 'S2_STAGE_DISC_PRESS',
 'S2_SCRU_SCRUB_LEVEL',
 'GAS_LIFT_HEAD_PRESS',
 'IN_CONT_CONT_VALVE',
 'IN_SEP_PRESS',
 'WH_PRESSURE']

from pandas.plotting import scatter_matrix
dataFrame = sampleEvent[colSet1]

scatter_matrix(dataFrame, alpha=0.2, figsize=(20, 20), diagonal='kde')
display()

# COMMAND ----------

from pandas.plotting import bootstrap_plot
import pandas as pd
dataFrame = sampleEvent['THROW_3_DISC_PRESS']
bootstrap_plot(dataFrame, size=50, samples=500, color='crimson')
display()

# COMMAND ----------

dataFrame = sampleEvent['THROW_2_DISC_PRESS']
bootstrap_plot(dataFrame, size=50, samples=500, color='crimson')
display()

# COMMAND ----------

engineSample50Set1.plot(subplots=True, layout=(3, 3), figsize=(30, 30), sharex=False);
display()

# COMMAND ----------

pd1 = spark.sql("select CODE,BF_SD_TYPE, COUNT(DISTINCT EVENT_ID) COUNT_EVENT from rc_data GROUP BY CODE,BF_SD_TYPE ")

# COMMAND ----------

display(pd1)

# COMMAND ----------

pd2 = spark.sql("select DAYTIME, THROW_2_DISC_TEMP  from rc_data where code = 'BEWF-ZZZ-F0110A' order by DAYTIME ")

# COMMAND ----------

pdDF = pd2.toPandas()

# COMMAND ----------

# MAGIC %sh
# MAGIC rm /databricks/driver/rc_sample_series_data.csv

# COMMAND ----------

pdDF.to_dense().to_csv("/databricks/driver/rc_sample_series_data.csv", index = False, sep=',', encoding='utf-8')

# COMMAND ----------

from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('/databricks/driver/rc_sample_series_data.csv', header=0)
series.hist()
##series.plot(subplots=True, layout=(3, 3), figsize=(30, 30), sharex=False)
#pyplot.show()
display()

# COMMAND ----------

print(series)