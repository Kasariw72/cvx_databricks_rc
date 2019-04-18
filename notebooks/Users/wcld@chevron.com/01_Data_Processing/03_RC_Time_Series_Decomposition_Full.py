# Databricks notebook source
# MAGIC %md # Remaining useful life (RUL) - Predictive Maintenance using LSTM 
# MAGIC 
# MAGIC Remaining useful life (RUL) is the useful life left on an asset at a particular time of operation. Its estimation is central to condition based maintenance and prognostics and health management. RUL is typically random and unknown, and as such it must be estimated from available sources of information such as the information obtained in condition and health monitoring. (Si X Wang and et al., 2010)
# MAGIC 
# MAGIC The concept of Remaining Useful Life (RUL) is utilised to predict life-span of components (of a service system) with the purpose of minimising catastrophic failure events in both manufacturing and service sectors.(Okoh C and et al., 2014)

# COMMAND ----------

# MAGIC %md # Time series decomposition
# MAGIC 
# MAGIC In the above data, a long term cyclic pattern seems to be non-existent. Also in theory, business cycles in traditional businesses are observed over a period of 3 or more years. Hence, we won’t include business cycles in this time series decomposition exercise. Also, we observe a overall increasing trend across years. We will build our model based on the following function:
# MAGIC 
# MAGIC $$
# MAGIC y_t = f(d_t, s_t, \varepsilon_t)
# MAGIC $$
# MAGIC 
# MAGIC where $d_t$ is the trend component, $s_t$ is the seasonal component and $\varepsilon_t$ is purely random noise.
# MAGIC 
# MAGIC ## Time series components
# MAGIC 
# MAGIC The fundamental idea of time series analysis is to decompose the original time series (sales, stock market trends, etc.) into several independent components. Typically, business time series are divided into the following four components:
# MAGIC 
# MAGIC <ol>
# MAGIC <li><strong>Trend</strong> – overall direction of the series i.e. upwards, downwards etc</li>
# MAGIC <li><strong>Seasonality</strong> – monthly or yearly patterns</li>
# MAGIC <li><strong>Cycle</strong> – long-term engine cycles, they usually come after 1 to 3 years</li>
# MAGIC <li><strong>Noise</strong> – irregular remainder left after extraction of all the components</li>
# MAGIC </ol>
# MAGIC 
# MAGIC Why bother decomposing the original / actual time series into components? It is much easier to forecast the individual regular patterns produced through decomposition of time series than the actual series. Let us have a quick look at our data series.

# COMMAND ----------

## Go to HDP Server to upload data file bz2 to the location and convert it to be parquet file to be sutiable for data analysis
## Command Lins on the following:
#spark-shell
##Create parquet file from json input file
#val path = "hdfs:///user/admin/data/data04_edited.csv.bz2"
#val sqlContext = new org.apache.spark.sql.SQLContext(sc)
#val df = sqlContext.read.option("delimiter", ",").option("header","true").option("inferSchema", "true").csv(path)
#df.write.format("parquet").save("hdfs:///user/admin/data/parquet/rc_lake.parquet")
# Open another terminal and run 
# tail -f jupyter_workspace/jupyter.log


# COMMAND ----------

#%load_ext autoreload
#%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# basic functionalities
import re
import os
import sys
import datetime
import itertools
import math 

# data transforamtion and manipulation
import pandas as pd
import pandas_datareader.data as web
import numpy as np
# prevent crazy long pandas prints
pd.options.display.max_columns = 16
pd.options.display.max_rows = 16
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)


# remove warnings
import warnings
warnings.filterwarnings('ignore')


# plotting and plot stying
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# jupyter wdgets
from ipywidgets import interactive, widgets, RadioButtons, ToggleButtons, Select, FloatSlider, FloatProgress
from IPython.display import set_matplotlib_formats, Image

# COMMAND ----------

plt.style.use('seaborn')
#sns.set_style("whitegrid", {'axes.grid' : False})
#set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"

# COMMAND ----------

#1 - import module
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
# import numpy
# import pandas

# COMMAND ----------

#2 - Create SparkContext
sc = SparkContext()
print(sc)

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "/home/admin/jupyter_workspace/RCPredictiveMA/"
sparkAppName = "RCDataAnlaysisFullSplit"
hdfsPathCSV = "hdfs:///user/admin/data/hbase_db/data06_20190125_Y*.csv"

# COMMAND ----------

#3 - Setup SparkSession(SparkSQL)
spark = (SparkSession
         .builder
         .appName(sparkAppName)
         .getOrCreate())
print(spark)

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

#5 - Print sample 5 rows of all variables and schema
#df.show(1)
#print("\n")
df.printSchema()

# COMMAND ----------

# df.sample(False, 0.05, 1234).toPandas()

# COMMAND ----------

df.columns

# COMMAND ----------

# df.describe().toPandas()

# COMMAND ----------

#6 - change column name
renamed_df = df.selectExpr("object_id as OBJECT_ID","OBJECT_CODE as CODE",
"time as DAYTIME","n_year as YEAR","n_month as MONTH",
"n_day as DAY","n_hour as HOUR",
"n_mm as MM","CURRENT_CYCLE as CYCLE",
"SD_TYPE as sd_type","DIGIT_SD_TYPE as DIGIT_SD_TYPE",
"ROUND(45 - S0_LUBE_OIL_PRESS, 7) AS RESI_LL_LUBE_OIL_PRESS", 
"ROUND(S2_LUBE_OIL_TEMP - 190,7) AS RESI_HH_LUBE_OIL_TEMP", 
"ROUND(820 - SB18_SPEED,7) AS RESI_LL_SPEED", 
"ROUND(SB18_SPEED - 1180,7) AS RESI_HH_SPEED", 
"ROUND(0.33 - SB19_VIBRA_ENGINE,7) AS RESI_LL_VIBRATION", 
"ROUND(SB19_VIBRA_ENGINE - 0.475,7) AS RESI_HH_VIBRATION", 
"ROUND(293.33 - S6_THROW_1_DISC_TEMP,7) AS RESI_HH_THROW1_DIS_TEMP", 
"ROUND(S8_THROW_1_SUC_TEMP - 190,7) AS RESI_HH_THROW1_SUC_TEMP", 
"S0_LUBE_OIL_PRESS AS LUBE_OIL_PRESS",
"S2_LUBE_OIL_TEMP AS LUBE_OIL_TEMP",
"S5_THROW_1_DISC_PRESS AS THROW_1_DISC_PRESS",
"S6_THROW_1_DISC_TEMP AS THROW_1_DISC_TEMP",
"S7_THROW_1_SUC_PRESS AS THROW_1_SUC_PRESS",
"S8_THROW_1_SUC_TEMP AS THROW_1_SUC_TEMP",
"S9_THROW_2_DISC_PRESS AS THROW_2_DISC_PRESS",
"S10_THROW_2_DISC_TEMP AS THROW_2_DISC_TEMP",
"S11_THROW_2_SUC_PRESS as THROW_2_SUC_PRESS",
"S12_THROW_2_SUC_TEMP AS THROW_2_SUC_TEMP",
"S13_THROW_3_DISC_PRESS AS  THROW_3_DISC_PRESS",
"S14_THROW_3_DISC_TEMP AS THROW_3_DISC_TEMP",
"S15_THROW_3_SUC_PRESS AS THROW_3_SUC_PRESS",
"S16_THROW_3_SUC_TEMP AS THROW_3_SUC_TEMP",
"S17_THROW_4_DISC_PRESS AS THROW_4_DISC_PRESS",
"S18_THROW_4_DISC_TEMP AS THROW_4_DISC_TEMP",
"S19_VIBRATION AS VIBRATION",
"SB1_CYL_1_TEMP AS CYL_1_TEMP",
"SB2_CYL_2_TEMP AS CYL_2_TEMP",
"SB3_CYL_3_TEMP AS CYL_3_TEMP",
"SB4_CYL_4_TEMP AS CYL_4_TEMP",
"SB5_CYL_5_TEMP AS CYL_5_TEMP",
"SB6_CYL_6_TEMP AS CYL_6_TEMP",
"SB7_CYL_7_TEMP AS CYL_7_TEMP",
"SB8_CYL_8_TEMP AS CYL_8_TEMP",
"SB9_CYL_9_TEMP AS CYL_9_TEMP",
"SB10_CYL_10_TEMP AS CYL_10_TEMP",
"SB11_CYL_11_TEMP AS CYL_11_TEMP",
"SB12_FUEL_GAS_PRESS AS FUEL_GAS_PRESS",
"SB14_LUBE_OIL_PRESS_ENGINE AS LUBE_OIL_PRESS_ENGINE",
"SB18_SPEED AS SPEED",
"SB19_VIBRA_ENGINE AS VIBRA_ENGINE",
"RUL as RUL",
"RUL*-1 as IN_RUL",
"EVENT_ID as EVENT_ID",
"LABEL1 as LABEL1",
"LABEL2 as LABEL2",
"BE_SD_TYPE as BE_SD_TYPE")


renamed_df.printSchema()

# COMMAND ----------

# #8 - sample data
# sample_df = renamed_df.sample(withReplacement=False, fraction=0.5, seed=1000)
print("sample_df count : " + str(renamed_df.count()))

# #This is just test and make the sample...

# COMMAND ----------

renamed_df.groupby("BE_SD_TYPE").count().toPandas()

# COMMAND ----------

renamed_df.where("BE_SD_TYPE='UCM'").groupby("LABEL1").count().toPandas()

# COMMAND ----------

renamed_df.columns

# COMMAND ----------

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
from pyspark.sql.functions import when

def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
        #pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNull(),pySparkDF[colName]).otherwise(0))
        med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
        pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(med[0]))
        
    return pySparkDF

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
 'FUEL_GAS_PRESS',
 'LUBE_OIL_PRESS_ENGINE',
 'SPEED',
 'VIBRA_ENGINE']
renamed_df = replaceByMedian(renamed_df,columnList)

# COMMAND ----------

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
from pyspark.sql.functions import when

def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
        med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
        pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(med[0]))
        print("Relaced [",colName,"] by ", med[0])
    return pySparkDF

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
 'FUEL_GAS_PRESS',
 'LUBE_OIL_PRESS_ENGINE',
 'SPEED',
 'VIBRA_ENGINE']
renamed_df = replaceByMedian(renamed_df,columnList)

# COMMAND ----------

## Handle Null in Cycle and Invert Order to plot graph

# COMMAND ----------

# renamed_df.describe().toPandas()

# COMMAND ----------

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data_05")
## Filter out the data during RC S/D
renamed_df = spark.sql("select * from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 0 ")
#renamed_df = spark.sql("select * from rc_data_04 where SD_TYPE is null")
#     .toPandas())
#renamed_df.describe().toPandas()
#plt.plot(df_tb1)

# COMMAND ----------

renamed_df3 = spark.sql("select EVENT_ID, CODE , COUNT(*) as num_steps from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")
renamed_df3.createOrReplaceTempView("tmp_step_event")
renamed_df3 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")

# COMMAND ----------

renamed_df.groupby("BE_SD_TYPE").count().toPandas()

# COMMAND ----------

renamed_df3.groupby("CODE").count().toPandas()

# COMMAND ----------

renamed_df3.groupby("EVENT_ID").count().toPandas()

# COMMAND ----------

# renamed_df3.toPandas().sort_values("num_steps", ascending=False)

def exportCSV(p_sd_type,p_rows):
    renamed_df3 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_data_05 where BE_SD_TYPE ='"+p_sd_type+"' and EVENT_ID <> 9999999 and RUL <> 9999999 GROUP BY  CODE, EVENT_ID")
    events = renamed_df3.toPandas()
    renamed_df3 = spark.sql("select EVENT_ID, CODE,  num_steps from tmp_step_event where  num_steps>="+str(p_rows)+" order by num_steps desc")
    
    for event_id in events.EVENT_ID:
      pathCSV = "/home/admin/jupyter_workspace/RCPredictiveMA/output/VER2/"+p_sd_type+"/"
      tdf = renamed_df.where(" RUL <>0 AND EVENT_ID="+str(event_id))
      numrows = tdf.count()
      rcCode = tdf.first().CODE[:4]
        
      tdf.toPandas().to_csv(pathCSV+rcCode+"_"+str(numrows)+"_ID_"+str(event_id)+".csv")
        
      print(event_id, ":", rcCode , " >> ", numrows)
      print("****** End Exporting Data Files at ", pathCSV)

# COMMAND ----------

# exportCSV("SPM",1440)
# exportCSV("SPM",2880)

# COMMAND ----------


# exportCSV("NMD",1440)
# exportCSV("NMD",2880)


# COMMAND ----------


# exportCSV("STB",1440)
# exportCSV("STB",2880)


# COMMAND ----------


# exportCSV("SCM",1440)
# exportCSV("SCM",2880)

# COMMAND ----------


# exportCSV("UCM",1440)
# exportCSV("UCM",2880)

# COMMAND ----------

renamed_df3 = spark.sql("select EVENT_ID, CODE,  num_steps from tmp_step_event where  num_steps>=1440 order by num_steps desc")


# COMMAND ----------

renamed_df3.describe().toPandas()

# COMMAND ----------

renamed_df3 = spark.sql("select OBJECT_ID, DAYTIME, SD_TYPE, YEAR, MONTH, DAY, HOUR, MM, RUL, CYCLE, EVENT_ID from rc_data_05 where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps>1440) order by YEAR, MONTH, DAY, HOUR, MM, RUL")


# COMMAND ----------

renamed_df3.show()

# COMMAND ----------

#10.1 - groupBy with count transactions before UCM event (minute time slot) by Event Id
renamed_df3.groupBy("EVENT_ID").count().toPandas()

# COMMAND ----------

# MAGIC %md There for we have 221 events of UCM for the 2 Remote Compressors between 2016 and 2018

# COMMAND ----------

# # Dealing with categorical string data type columns
# object_columns_df = (renamed_df.toPandas()).select_dtypes(include=['object']).copy()
# # Print totoal Null Value of the DF in object columns
# print(object_columns_df.isnull().values.sum())

# COMMAND ----------

# # Dealing with null value of each numeric data type columns
# int_columns_df = (renamed_df2.toPandas()).select_dtypes(include=['double']).copy()
# # Print totoal Null Value of the DF in object columns
# print(int_columns_df.isnull().values.sum())
# print(int_columns_df.isnull().sum())

# COMMAND ----------

print(spark.catalog.listTables())

# COMMAND ----------

#df_simple = spark.sql("select daytime, cycle, S1 from rc_data_04 where cycle <> 9999999 and obj_id=19 and year=2018 and month=1 and day between 1 and 31 order by cycle")
## percentile_approx(x, 0.5)

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
    +"AVG( FUEL_GAS_PRESS) AS  FUEL_GAS_PRESS,"
    +"AVG( LUBE_OIL_PRESS_ENGINE) AS  LUBE_OIL_PRESS_ENGINE,"
    +"AVG( SPEED) AS  SPEED,"
    +"AVG( VIBRA_ENGINE) AS  VIBRA_ENGINE "
    + " FROM rc_data_05 where YEAR between 2016 and 2018 "
    + " AND RUL between "+ str(P_RUL_FROM) +" AND "+ str(P_RUL_TO)  +" AND BE_SD_TYPE = '"+ P_SD_TYPE +"' " 
    + " AND EVENT_ID <> 9999999 and CYCLE <> 9999999 and RUL<> 0 "
    + " AND EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event WHERE num_steps > "+ str(P_MINIMUM_CYCLE)+") "
    + " GROUP BY IN_RUL order by IN_RUL"
                         )
    return sparkDataFrame




# COMMAND ----------

# renamed_df5 = spark.sql(" select IN_RUL, "
# +"AVG(RESI_LL_LUBE_OIL_PRESS) AS RESI_LL_LUBE_OIL_PRESS,"
# +"AVG( RESI_HH_LUBE_OIL_TEMP) AS  RESI_HH_LUBE_OIL_TEMP,"
# +"AVG( RESI_LL_SPEED) AS  RESI_LL_SPEED,"
# +"AVG( RESI_HH_SPEED) AS  RESI_HH_SPEED,"
# +"AVG( RESI_LL_VIBRATION) AS  RESI_LL_VIBRATION,"
# +"AVG( RESI_HH_VIBRATION) AS  RESI_HH_VIBRATION,"
# +"AVG( RESI_HH_THROW1_DIS_TEMP) AS  RESI_HH_THROW1_DIS_TEMP,"
# +"AVG( RESI_HH_THROW1_SUC_TEMP) AS  RESI_HH_THROW1_SUC_TEMP,"
# +"AVG( LUBE_OIL_PRESS) AS  LUBE_OIL_PRESS,"
# +"AVG( LUBE_OIL_TEMP) AS  LUBE_OIL_TEMP,"
# +"AVG( THROW_1_DISC_PRESS) AS  THROW_1_DISC_PRESS,"
# +"AVG( THROW_1_DISC_TEMP) AS  THROW_1_DISC_TEMP,"
# +"AVG( THROW_1_SUC_PRESS) AS  THROW_1_SUC_PRESS,"
# +"AVG( THROW_1_SUC_TEMP) AS  THROW_1_SUC_TEMP,"
# +"AVG( THROW_2_DISC_PRESS) AS  THROW_2_DISC_PRESS,"
# +"AVG( THROW_2_DISC_TEMP) AS  THROW_2_DISC_TEMP,"
# +"AVG( THROW_2_SUC_TEMP) AS  THROW_2_SUC_TEMP,"
# +"AVG( THROW_3_DISC_PRESS) AS  THROW_3_DISC_PRESS,"
# +"AVG( THROW_3_DISC_TEMP) AS  THROW_3_DISC_TEMP,"
# +"AVG( THROW_3_SUC_PRESS) AS  THROW_3_SUC_PRESS,"
# +"AVG( THROW_3_SUC_TEMP) AS  THROW_3_SUC_TEMP,"
# +"AVG( THROW_4_DISC_PRESS) AS  THROW_4_DISC_PRESS,"
# +"AVG( THROW_4_DISC_TEMP) AS  THROW_4_DISC_TEMP,"
# +"AVG( VIBRATION) AS  VIBRATION,"
# +"AVG( CYL_1_TEMP) AS  CYL_1_TEMP,"
# +"AVG( CYL_2_TEMP) AS  CYL_2_TEMP,"
# +"AVG( CYL_3_TEMP) AS  CYL_3_TEMP,"
# +"AVG( CYL_4_TEMP) AS  CYL_4_TEMP,"
# +"AVG( CYL_5_TEMP) AS  CYL_5_TEMP,"
# +"AVG( CYL_6_TEMP) AS  CYL_6_TEMP,"
# +"AVG( CYL_7_TEMP) AS  CYL_7_TEMP,"
# +"AVG( CYL_8_TEMP) AS  CYL_8_TEMP,"
# +"AVG( CYL_9_TEMP) AS  CYL_9_TEMP,"
# +"AVG( CYL_10_TEMP) AS  CYL_10_TEMP,"
# +"AVG( CYL_11_TEMP) AS  CYL_11_TEMP,"
# +"AVG( FUEL_GAS_PRESS) AS  FUEL_GAS_PRESS,"
# +"AVG( LUBE_OIL_PRESS_ENGINE) AS  LUBE_OIL_PRESS_ENGINE,"
# +"AVG( SPEED) AS  SPEED,"
# +"AVG( VIBRA_ENGINE) AS  VIBRA_ENGINE "
# + " from rc_data_05 where "
# + " IN_RUL between -1440 and -1 and BE_SD_TYPE = 'UCM' " 
# + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 "
# + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 1440) group by IN_RUL "
# + " order by IN_RUL ")

# COMMAND ----------

# (renamed_df3.columns)

# COMMAND ----------

# renamed_df3.count()

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
 'FUEL_GAS_PRESS',
 'LUBE_OIL_PRESS_ENGINE',
 'SPEED',
 'VIBRA_ENGINE']

#df_simple = replaceByMedian(df_simple ,columnList)

# print(df_simple.columns)
#df_simple.toPandas()
## pd_df_1 =  pd_df_1.withColumn('curr_rul',pd_df_1['curr_rul'])                         
#df_simple.sort_values('curr_rul')

P_SD_TYPES = ["NMD","SCM","STB","SPM","UCM"]
P_START_RUL = 1
P_END_RUL = 1440
P_MINIMUM_RUL = 1440
P_CUR_SD_TYPE = "NMD"

renamed_df = queryAVGSensorByInRUL(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) 
pd_df_1 = (renamed_df.toPandas())

# COMMAND ----------

# df_simple.describe().toPandas()

# COMMAND ----------

# pd_df_1.sort_values('IN_RUL',ascending=True)
# ##pd_df_1.describe()

# COMMAND ----------

# simple line plot
plt.style.use('seaborn')
#sns.set_style("whitegrid", {'axes.grid' : False})
#set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = b"\usepackage{subdepth}, \usepackage{type1cm}"

yList = pd_df_1['LUBE_OIL_PRESS']
xList = pd_df_1['IN_RUL']
##plt.style.use('seaborn')
plt.plot(xList, yList)
plt.title('Sensor Plot', fontsize=24)
plt.ylabel('Average Engine Vibration')
plt.xlabel('Time Step')
#plt.xlim(vList.count() * - 0.1, vList.count() * 1.1)
plt.savefig(path+'/img/AVG_ENG_VIBRA.png')
plt.show()

# COMMAND ----------

# MAGIC %md ## Trend
# MAGIC 
# MAGIC From the preliminary plots so far it is obvious that there is some kind of increasing <strong>trend</strong> in the series along with seasonal variation. Since stationarity is a vital assumption we need to verify if our time series follows a stationary process or not. We can do so by
# MAGIC 
# MAGIC <ol>
# MAGIC <li><strong>Plots</strong>: review the time series plot of our data and visually check if there are any obvious trends or seasonality</li>
# MAGIC <li><strong>Statistical tests</strong>: use statistical tests to check if the expectations of stationarity are met or have been violated.</li>
# MAGIC </ol>
# MAGIC 
# MAGIC ### Moving averages over time
# MAGIC 
# MAGIC One way to identify a trend pattern is to use moving averages over a specific window of past observations. This smoothes the curve by averaging adjacent values over the specified time horizon (window).

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
    plt.show(block=False)
    return

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
    plt.show()

    return

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

    axes[0][1].set_xlabel("Cycle");
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
    plt.show()

    return

# COMMAND ----------

windowtime = 15

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

# COMMAND ----------

P_SD_TYPES = ["NMD","SCM","STB","SPM","UCM"]

P_START_RUL = 1
P_END_RUL = 2880
P_MINIMUM_RUL = 2879
P_CUR_SD_TYPE = "NMD"

selectedCols = columnList 

for P_CUR_SD_TYPE in P_SD_TYPES: 
    renamed_df = queryAVGSensorByInRUL(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) 
    pd_df_1 = (renamed_df.toPandas())
    drawGraphBySDType(P_CUR_SD_TYPE)
    for colName in selectedCols:
        #pd_df_1[colName].head()
        plot_rolling_average(pd_df_1['IN_RUL'], pd_df_1[colName],windowtime,P_CUR_SD_TYPE)

print("**Finished**")

# COMMAND ----------

# ## Print Single Sensor Moving Average Per Invese RUL (Minute)
# # selectedCols = columnList 

    
# for P_CUR_SD_TYPE in P_SD_TYPES: 
#     renamed_df = queryAVGSensorByInRUL(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) 
#     pd_df_1 = (renamed_df.toPandas())
    
    
    
    


# COMMAND ----------

# MAGIC %md ## Seasonality
# MAGIC People tend to go on vacation mainly during summer holidays. That is, at some time periods during the year people tend to use aircrafts more frequently. We could check this hypothesis of a seasonal effect by 

# COMMAND ----------

def queryAVGSensorByEventId(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) :
    df_simple2 = spark.sql(" select  IN_RUL , EVENT_ID ,"
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
                            +"AVG( FUEL_GAS_PRESS) AS  FUEL_GAS_PRESS,"
                            +"AVG( LUBE_OIL_PRESS_ENGINE) AS  LUBE_OIL_PRESS_ENGINE,"
                            +"AVG( SPEED) AS  SPEED,"
                            +"AVG( VIBRA_ENGINE) AS  VIBRA_ENGINE "
                          + " from rc_data_05 where "
                          + " RUL between "+ str(P_START_RUL) +" and "+str(P_END_RUL) 
                          + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 and RUL<>0 "
                          + " and BE_SD_TYPE ='" + P_CUR_SD_TYPE+ "'"
                          + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > "+str(P_MINIMUM_RUL)+" )"
                          + " group by IN_RUL , EVENT_ID "
                          + " order by IN_RUL , EVENT_ID ")
    return df_simple2


# COMMAND ----------

# df_piv_line.groupby("YEAR").count().toPandas()

# COMMAND ----------


# pd_df = (df_piv_line.toPandas())


# COMMAND ----------

# pd_df.describe()

# COMMAND ----------

# pd_df.to_csv('/home/admin/data/rc/descibe_data_05_clean.csv')

# COMMAND ----------

# create line plot
def plot_line_graph(pivot_df, parameterName,sd_type,rul_start,rul_end):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pivot_df.plot(colormap='jet');
    plt.title("Line Plot of Sensor: "+parameterName+"("+sd_type+")"+str(rul_start*-1)+" to "+str(rul_end*-1), fontsize=18)
    plt.ylabel('Sensor Value of '+parameterName)
    #plt.xLabel("RUL:"+str(rul_start*-1)+" to "+str(rul_end*-1)+" time steps (minute)")
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(path+'img/'+sd_type+'_SEL_'+timestamp+"_"+parameterName+"_"+str(rul_start*-1)+"_"+str(rul_end*-1)+'.png')
    ##plt.show()
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
 'FUEL_GAS_PRESS',
 'LUBE_OIL_PRESS_ENGINE',
 'SPEED',
 'VIBRA_ENGINE']

P_SD_TYPES = ["NMD","SCM","STB","SPM","UCM"]
P_START_RUL = 1
P_END_RUL = 2440
P_MINIMUM_RUL = 1440
P_CUR_SD_TYPE = "NMD"
selectedCols = selectedCols3 

for P_CUR_SD_TYPE in P_SD_TYPES: 
    renamed_df = queryAVGSensorByEventId(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL) 
    pd_df_1 = (renamed_df.toPandas())
    for colName in selectedCols3:
        df_piv_box = pd_df_1.pivot_table(index=['IN_RUL'], columns='EVENT_ID',values= colName , aggfunc='mean')
        plot_line_graph(df_piv_box,colName,P_CUR_SD_TYPE,P_START_RUL,P_END_RUL)

print("**Finish drawing graph.....**")


# COMMAND ----------

# create nice axes names
month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')
# reshape date

def plot_boxPlot(df_piv_box, parameterName):
    ##print(df_piv_box)
    # create a box plot
    fig, ax = plt.subplots()
    df_piv_box.plot(ax=ax, kind='box')
    ax.set_title('Seasonal Effect per Month', fontsize=24)
    ax.set_xlabel('Month')
    ax.set_ylabel(parameterName)
    ax.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.savefig(path+'img/seasonal_effect_boxplot_400to1440_'+parameterName+'.png')
    plt.show()
    # plot heatmap
    sns.heatmap(df_piv_box, annot=False)
    return

# COMMAND ----------

## Print Single Sensor Moving Average Per Invese RUL (Minute)
selectedCols2 = ['RESI_LL_LUBE_OIL_PRESS',
       'RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_SPEED', 'RESI_HH_SPEED',
       'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION', 'RESI_HH_THROW1_DIS_TEMP',
       'RESI_HH_THROW1_SUC_TEMP', 'AVG_LUBE_OIL_PRESS', 'AVG_LUBE_OIL_TEMP',
       'AVG_THROW1_SUC_TEMP', 'AVG_ENG_SPEED', 'AVG_ENG_VIBRA', 'S3', 'S4',
       'S5', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
       'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26',
       'S27', 'S28', 'S29']
    
for colName in selectedCols2:
    df_piv_line = pd_df.pivot_table(index=['MONTH'],columns='YEAR',values=colName, aggfunc='mean')
    plot_colormap_season(df_piv_line,colName)

# COMMAND ----------


for colName in selectedCols2:
    df_piv_box = pd_df.pivot_table(index=['MONTH'], columns='YEAR',values= colName , aggfunc='mean')
    plot_boxPlot(df_piv_box,colName)
    
print("*** End **** ")

# COMMAND ----------


for colName in selectedCols2:
    df_piv_box = pd_df.pivot_table(index=['MONTH'], columns='RUL',values= colName , aggfunc='mean')
    plot_boxPlot(df_piv_box,colName)
    
print("*** End **** ")

# COMMAND ----------

df_simple2 = spark.sql(" select IN_RUL , EVENT_ID , avg(RESI_LL_LUBE_OIL_PRESS) as RESI_LL_LUBE_OIL_PRESS "
                      + ",avg(RESI_HH_LUBE_OIL_TEMP) as RESI_HH_LUBE_OIL_TEMP "
                      + ",avg(RESI_LL_SPEED) as RESI_LL_SPEED "
                      + ",avg(RESI_HH_SPEED) as RESI_HH_SPEED "
                      + ",avg(RESI_LL_VIBRATION) as RESI_LL_VIBRATION "
                      + ",avg(RESI_HH_VIBRATION) as RESI_HH_VIBRATION "
                      + ",avg(RESI_HH_THROW1_DIS_TEMP) as RESI_HH_THROW1_DIS_TEMP"
                      + ",avg(RESI_HH_THROW1_SUC_TEMP) as RESI_HH_THROW1_SUC_TEMP "
                      + ",avg(S1) as AVG_LUBE_OIL_PRESS"
                      + ",avg(S2) as AVG_LUBE_OIL_TEMP  "
                      + ",avg(S6) as AVG_THROW1_SUC_TEMP "
                      + ",avg(S32) as AVG_ENG_SPEED"
                      + ",avg(S33) as AVG_ENG_VIBRA "
                      + ",avg(S3) as S3 "
                      + ",avg(S4) as S4 "
                      + ",avg(S5) as S5 "
                      + ",avg(S7) as S7 "
                      + ",avg(S8) as S8 "
                      + ",avg(S9) as S9 "
                      + ",avg(S10) as S10 "
                      + ",avg(S11) as S11 "
                      + ",avg(S12) as S12 "
                      + ",avg(S13) as S13 "
                      + ",avg(S14) as S14 "
                      + ",avg(S15) as S15 "
                      + ",avg(S16) as S16 "
                      + ",avg(S17) as S17 "
                      + ",avg(S18) as S18 "
                      + ",avg(S19) as S19 "
                      + ",avg(S20) as S20 "
                      + ",avg(S21) as S21 "
                      + ",avg(S22) as S22 "
                      + ",avg(S23) as S23 "
                      + ",avg(S24) as S24 "
                      + ",avg(S25) as S25 "
                      + ",avg(S26) as S26 "
                      + ",avg(S27) as S27 "
                      + ",avg(S28) as S28 "
                      + ",avg(S29) as S29 "
                      + " from rc_data_05 where BE_SD_TYPE = 'UCM' "
                      + " and RUL between 400 and 600 " 
                      + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 "
                      + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps <= 1440)"
                      + " group by IN_RUL , EVENT_ID "
                      + " order by IN_RUL , EVENT_ID ")

# COMMAND ----------

# df_simple2.show()

# COMMAND ----------

#df_simple2.head()
pd_df_2 = (df_simple2.toPandas())

# COMMAND ----------

pd_df_2.columns

# COMMAND ----------

# pd_history_event = df_simple2.groupBy("event_id").count()
# pd_history_event.show() ## Show in style Spark Data Frame

# COMMAND ----------

# pd_df_2.sort_values('IN_RUL',ascending=True)

# COMMAND ----------

# pd_df_2.describe()

# COMMAND ----------

# import numpy as np
# import matplotlib.pyplot as plt

# def plot_graph(rul,x,y,a,b, w):
#     # Fixing random state for reproducibility
#     fig, axs = plt.subplots(2, 1)
    
#     axs[0].plot(rul, x, label = x.name)
#     #axs[0].set_xlim(0, 2)
#     axs[0].set_xlabel('Inverse Remaining Useful Life (Minute)')
#     axs[0].set_ylabel('s1 s2 s3 24')
#     axs[0].grid(True)
    
#     cxy, f = axs[1].cohere(x, ,y, a , b)
#     axs[1].set_ylabel('coherence')
    
#     fig.tight_layout()
#     plt.show()

# COMMAND ----------

# create line plot
def plot_line_graph(pivot_df, parameterName):
    pivot_df.plot(colormap='jet');
    plt.title(' Line Plot of Sensors : '+parameterName, fontsize=22)
    plt.ylabel('Sensor Value')
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(path+'img/seasonal_effect_lines_less_1440times_'+parameterName+'.png')
    plt.show()
    return

selectedCols3 = ['RESI_LL_LUBE_OIL_PRESS', 'RESI_HH_LUBE_OIL_TEMP',
       'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION',
       'RESI_HH_VIBRATION', 'RESI_HH_THROW1_DIS_TEMP',
       'RESI_HH_THROW1_SUC_TEMP', 'AVG_LUBE_OIL_PRESS', 'AVG_LUBE_OIL_TEMP',
       'AVG_THROW1_SUC_TEMP', 'AVG_ENG_SPEED', 'AVG_ENG_VIBRA', 'S3', 'S4',
       'S5', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
       'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26',
       'S27', 'S28', 'S29']

selectedCols3 = ['RESI_LL_LUBE_OIL_PRESS', 'RESI_HH_LUBE_OIL_TEMP',
       'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION',
       'RESI_HH_VIBRATION', 'RESI_HH_THROW1_DIS_TEMP',
       'RESI_HH_THROW1_SUC_TEMP', 'AVG_LUBE_OIL_PRESS', 'AVG_LUBE_OIL_TEMP',
       'AVG_THROW1_SUC_TEMP', 'AVG_ENG_SPEED', 'AVG_ENG_VIBRA', 'S3', 'S4',
       'S5', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
       'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26',
       'S27', 'S28', 'S29']
    
for colName in selectedCols3:
    df_piv_box = pd_df_2.pivot_table(index=['IN_RUL'], columns='EVENT_ID',values= colName , aggfunc='mean')
    #df_piv_box
    plot_line_graph(df_piv_box,colName)
    

# COMMAND ----------

# #10.2 - groupBy with count
df_simple2.groupBy("EVENT_ID").count().toPandas().sort_values('count',ascending=True).describe()
df_simple2.groupBy("EVENT_ID").count().toPandas().sort_values('count',ascending=False)

df_tmp = df_simple2.groupBy("EVENT_ID").count().toPandas().sort_values('count',ascending=False)
print(df_simple2.columns)
print(df_tmp)

# COMMAND ----------

print(df_simple2.columns)

# COMMAND ----------

# for colName in selectedCols2:
#     df_piv_box = pd_df.pivot_table(index=['curr_rul'], columns='Month',values= colName , aggfunc='mean')
#     plot_boxPlot(df_piv_box,colName)

# COMMAND ----------

# sensor_x = 'AVG_LUBE_OIL_PRESS'

# #df_piv_line = spark.sql(" select daytime, Curr_RUL, avg("+sensor_x+") as "+sensor_x+" from rc_data_04 where cycle <> 9999999 and Curr_RUL between 0 and 500 group by Month, CURR_RUL order by Month, Curr_RUL ")
# #df_simple = spark.sql("select daytime, cycle, S1 from rc_data_04 where cycle <> 9999999 and obj_id=19 and year=2018 and month=1 and day between 1 and 31 order by cycle")
# df_piv_line = spark.sql(" select daytime, Month, curr_rul "
#                       + ",avg(S1) as AVG_LUBE_OIL_PRESS"
#                       + ",avg(S2) as AVG_LUBE_OIL_TEMP  "
#                       + ",avg(S6) as AVG_THROW1_SUC_TEMP "
#                       + ",avg(S32) as AVG_ENG_SPEED"
#                       + ",avg(S33) as AVG_ENG_VIBRA "
#                       + " from rc_data_04 where year between 2016 and 2018 and curr_rul between 0 and 600 group by daytime, Month, curr_rul ")

# pd_df = (df_piv_line.toPandas())

# COMMAND ----------

# df_piv_line = pd_df.pivot_table(index=['Month'],columns='curr_rul',values=sensor_x, aggfunc='mean')

# #df_piv_line.toPandas()

# COMMAND ----------

#pd_df['month'] = pd_df.index.month
#pd_df['year'] = pd_df.index.year

# create nice axes names
#month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')
#print(month_names)
#df_piv_line = pd_df.reindex(index=month_names)
# reshape data using 'Year' as index and 'Month' as column
#df_piv_line = pd_df.pivot(index='Month', columns='Year', values='S1')

#df_piv_line = df_piv.reindex(index=month_names)
#print(df_piv_line)

# COMMAND ----------

# # create line plot
# df_piv_line.plot(colormap='jet');
# plt.title('Seasonal Effect per Month', fontsize=24)
# plt.ylabel('Sensor Value')
# #plt.xlabel('Month')
# plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
# plt.savefig(path+'img/seasonal_effect_lines.png')
# plt.show()

# COMMAND ----------

# # create nice axes names
# month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')

# # reshape date
# df_piv_box = pd_df.pivot_table(index=['curr_rul'], columns='Month',values= sensor_x, aggfunc='mean')
# print(df_piv_box)

# # reindex pivot table with 'month_names'
# #df_piv_box2 = df_piv_box.reindex(columns=month_names)
# #print('/n')
# #print(df_piv_box2)

# # create a box plot
# fig, ax = plt.subplots()
# df_piv_box.plot(ax=ax, kind='box')
# ax.set_title('Seasonal Effect per Month', fontsize=24)
# ax.set_xlabel('Month')
# ax.set_ylabel('S1')
# ax.xaxis.set_ticks_position('bottom')
# fig.tight_layout()
# plt.savefig(path+'img/seasonal_effect_boxplot.png')
# plt.show()

# COMMAND ----------

# # plot heatmap
# sns.heatmap(df_piv_box, annot=False)

# COMMAND ----------

# sns.heatmap(df_piv_line, annot=False)

# COMMAND ----------

# MAGIC %md ## Noise
# MAGIC 
# MAGIC To understand the underlying pattern in the number of international airline passengers, we assume a <strong>multiplicative</strong> time series decomposition model with the following equation
# MAGIC 
# MAGIC $$
# MAGIC y_t = d_t \cdot s_t \cdot \varepsilon_t
# MAGIC $$
# MAGIC 
# MAGIC The <code>statsmodels</code> library grants access to predefined methods for decomposing time series automatically. Using the code below calls the <code>seasonal_decomposition</code> method which will return the original time series, the trend component, seasonality component. A specification of wether to apply an additive or multiplicative model is required.
# MAGIC 
# MAGIC <strong>However</strong>, be aware that plain vanilla decomposition models like these are rarely used for forecasting. Their primary purpose is to understand underlying patterns in temporal data to use in more sophisticated analysis like Holt-Winters seasonal method or ARIMA.

# COMMAND ----------

# # statistical modeling libraries
# from statsmodels.tsa.seasonal import seasonal_decompose
# import statsmodels.formula.api as smf
# import statsmodels.tsa.api as smt
# import statsmodels.api as sm
# import scipy.stats as scs
# #from arch import arch_model

# COMMAND ----------



# COMMAND ----------

# sensor_x = 'AVG_LUBE_OIL_PRESS'

# #df_piv_line = spark.sql(" select daytime, Curr_RUL, avg("+sensor_x+") as "+sensor_x+" from rc_data_04 where cycle <> 9999999 and Curr_RUL between 0 and 500 group by Month, CURR_RUL order by Month, Curr_RUL ")
# #df_simple = spark.sql("select daytime, cycle, S1 from rc_data_04 where cycle <> 9999999 and obj_id=19 and year=2018 and month=1 and day between 1 and 31 order by cycle")
# df_spark_2 = spark.sql(" select curr_rul "
#                       + ",avg(S1) as AVG_LUBE_OIL_PRESS"
#                       + ",avg(S2) as AVG_LUBE_OIL_TEMP  "
#                       + ",avg(S6) as AVG_THROW1_SUC_TEMP "
#                       + ",avg(S32) as AVG_ENG_SPEED"
#                       + ",avg(S33) as AVG_ENG_VIBRA "
#                       + " from rc_data_04 where year between 2016 and 2018 and curr_rul between 0 and 500 group by curr_rul ")

# #AVG_LUBE_OIL_PRESS2 = df_spark_2['AVG_LUBE_OIL_PRESS']

# # reshape date
# #df_piv_box = pd_df.pivot_table(index=['Curr_RUL'], columns='Month',values= sensor_x, aggfunc='mean')
# #print(df_piv_box)

# pd_df2 = (df_spark_2.toPandas())



# COMMAND ----------

# pd_df2.index = pd.DatetimeIndex(freq="w",start=0, periods=501)
# pd_df2.head(10)

# import statsmodels.api as sm
# decomp = sm.tsa.seasonal_decompose(pd_df2[0])
# decomp.plot()
# plt.show()

# COMMAND ----------

# # multiplicative seasonal decomposition
# #print("Y")
# #print(AVG_ENG_VIBRA)
# #y_mean = y.mean()
# #print("Y - Mean : ", y_mean)
# decomp = seasonal_decompose(pd_df2, model='multiplicative')
# decomp.plot();
# plt.savefig(path+'img/decomposition.png')
# plt.show()

# COMMAND ----------

# MAGIC %md <strong>Inference</strong>
# MAGIC <ul>
# MAGIC <li><strong>Trend</strong> - The 12 month moving average calculated earlier looks quite alike. Hence, we could have used linear regression to estimate the trend in this data.</li>
# MAGIC <li><strong>Seasonality</strong> - The seasonal plot displays a consistent month-to-month pattern. The monthly seasonal components are average values for a month after trend removal</li>
# MAGIC <li><strong>Noise</strong> - is the residual series left after removing the trend and seasonality components</li>
# MAGIC </ul>
# MAGIC 
# MAGIC Alternatively and for comparison, an <strong>additive</strong> model would be
# MAGIC 
# MAGIC $$
# MAGIC y_t = d_t + s_t + \varepsilon_t
# MAGIC $$

# COMMAND ----------

# # additive seasonal decomposition
# print(y)
# decomp = seasonal_decompose(y, model='additive')
# decomp.plot();
# plt.savefig(path+'img/decomp_additive.png')
# plt.show()

# COMMAND ----------

# MAGIC %md <strong>Inference</strong>
# MAGIC <ul>
# MAGIC <li>Difference in the scale of the decomposed time series when compared to the multiplicative model</li>
# MAGIC <li>There is still a pattern left in the residual series</li>
# MAGIC <li>The high residual values indicate that the application of a multiplicative model is preferable</li>
# MAGIC </ul>
# MAGIC 
# MAGIC <strong>Best Practice</strong>
# MAGIC <ul>
# MAGIC <li>When the data contains negative values, e.g. a data series on average temperatures, an additive model should be used</li>
# MAGIC </ul>