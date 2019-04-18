# Databricks notebook source
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
sparkAppName = "RCDataAnlaysisUCM"
#hdfsPathCSV = "hdfs:///user/admin/data/hbase_db/data05_20190115_ALL_CLEAN.csv"
hdfsPathClenMedCSV = "/user/admin/data/hbase_db/cleanmed/RC_UCM_*.csv"

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
        .csv(hdfsPathClenMedCSV)
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
renamed_df = df.selectExpr(
"OBJECT_ID AS  OBJECT_ID",
"CODE AS   CODE",
"DAYTIME AS   DAYTIME",
"YEAR AS   YEAR",
"MONTH AS   MONTH",
"DAY AS   DAY",
"HOUR AS   HOUR",
"MM AS   MM",
"CYCLE AS   CYCLE",
"sd_type AS   SD_TYPE",
"DIGIT_SD_TYPE AS   DIGIT_SD_TYPE",
"RESI_LL_LUBE_OIL_PRESS AS  RESI_LL_LUBE_OIL_PRESS",
"RESI_HH_LUBE_OIL_TEMP AS  RESI_HH_LUBE_OIL_TEMP",
"RESI_LL_SPEED AS  RESI_LL_SPEED",
"RESI_HH_SPEED AS  RESI_HH_SPEED",
"RESI_LL_VIBRATION AS  RESI_LL_VIBRATION",
"RESI_HH_VIBRATION AS  RESI_HH_VIBRATION",
"RESI_HH_THROW1_DIS_TEMP AS  RESI_HH_THROW1_DIS_TEMP",
"RESI_HH_THROW1_SUC_TEMP AS  RESI_HH_THROW1_SUC_TEMP",
"S1 AS LUBE_OIL_PRESS",
"S2 AS LUBE_OIL_TEMP",
"S3 AS THROW_1_DISC_PRESS",
"S4 AS THROW_1_DISC_TEMP",
"S5 AS THROW_1_SUC_PRESS",
"S6 AS THROW_1_SUC_TEMP",
"S7 AS THROW_2_DISC_PRESS",
"S8 AS THROW_2_DISC_TEMP",
"S9 AS THROW_2_SUC_TEMP",
"S10 AS  THROW_3_DISC_PRESS",
"S11 AS THROW_3_DISC_TEMP",
"S12 AS THROW_3_SUC_PRESS",
"S13 AS THROW_3_SUC_TEMP",
"S14 AS THROW_4_DISC_PRESS",
"S15 AS THROW_4_DISC_TEMP",
"S16 AS VIBRATION",
"S17 AS CYL_1_TEMP",
"S18 AS CYL_2_TEMP",
"S19 AS CYL_3_TEMP",
"S20 AS CYL_4_TEMP",
"S21 AS CYL_5_TEMP",
"S22 AS CYL_6_TEMP",
"S23 AS CYL_7_TEMP",
"S24 AS CYL_8_TEMP",
"S25 AS CYL_9_TEMP",
"S26 AS CYL_10_TEMP",
"S27 AS CYL_11_TEMP",
"S28 AS FUEL_GAS_PRESS",
"S29 AS LUBE_OIL_PRESS_ENGINE",
"S32 AS SPEED",
"S33 AS VIBRA_ENGINE",
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

renamed_df.columns

# COMMAND ----------

# #9 - union and intersect
# sample1_df = renamed_df.sample(withReplacement=False, fraction=0.5, seed=25)
# sample2_df = renamed_df.sample(withReplacement=False, fraction=0.5, seed=50)

# union_df = sample1_df.union(sample2_df)
# intersected_df = sample1_df.intersect(sample2_df)

# print("sample1_df count : " + str(sample1_df.count()))
# print("sample2_df count : " + str(sample2_df.count()))
# print("union_df count : " + str(union_df.count()))
# print("intersected_df count : " + str(intersected_df.count()))

# COMMAND ----------

# # #10.1 - groupBy with count
renamed_df.groupBy("BE_SD_TYPE").count().toPandas()

# COMMAND ----------

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data_06")
## Filter out the data during RC S/D
renamed_df = spark.sql("select * from rc_data_06 ")


# COMMAND ----------

renamed_df2 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps FROM rc_data_06  GROUP BY  CODE, EVENT_ID ")
renamed_df2.createOrReplaceTempView("tmp_step_event")

# COMMAND ----------

# renamed_df3.toPandas().sort_values("num_steps", ascending=False)

# COMMAND ----------

renamed_df2 = spark.sql("select EVENT_ID, CODE,  num_steps from tmp_step_event where  num_steps<2880 order by num_steps desc")
renamed_df3 = spark.sql("select EVENT_ID, CODE,  num_steps from tmp_step_event where  num_steps>=2880 order by num_steps desc")

# COMMAND ----------

renamed_df2.describe().toPandas()

# COMMAND ----------

events = renamed_df2.toPandas()

# COMMAND ----------

for event_id in events.EVENT_ID:
    pathCSV = "/home/admin/jupyter_workspace/RCPredictiveMA/output/"
    tdf = renamed_df.where(" RUL <>0 AND EVENT_ID="+str(event_id))
    
    numrows = tdf.count()
    rcCode = tdf.first().CODE
    tdf.toPandas().to_csv(pathCSV+rcCode+"_"+str(numrows)+"_ID_"+str(event_id)+".csv")
    print(event_id, ":", rcCode , " >> ", numrows)
    
print("****** End Exporting Data Files at ", pathCSV)


# COMMAND ----------

events = renamed_df3.toPandas()

for event_id in events.EVENT_ID:
    pathCSV = "/home/admin/jupyter_workspace/RCPredictiveMA/output/"
    tdf = renamed_df.where(" RUL <>0 AND EVENT_ID="+str(event_id))
    
    numrows = tdf.count()
    rcCode = tdf.first().CODE
    tdf.toPandas().to_csv(pathCSV+"LONG_"+rcCode+"_"+str(numrows)+"_ID_"+str(event_id)+".csv")
    print(event_id, ":", rcCode , " >> ", numrows)
    
print("****** End Exporting Data Files at ", pathCSV)


# COMMAND ----------

renamed_df3.describe().toPandas()

# COMMAND ----------

# pd1 = renamed_df2.toPandas()

# g = pd1.groupby('CODE').Zscore
# n = g.ngroups
# fig, axes = plt.subplots(n // 2, 2, figsize=(6, 6), sharex=True, sharey=True)

# for i, (name, group) in enumerate(g):
#     r, c = i // 2, i % 2
#     a1 = axes[r, c]
#     a2 = a1.twinx()
#     group.plot.hist(ax=a2, alpha=.3)
#     group.plot.kde(title=name, ax=a1, c='r')
# fig.tight_layout()

# COMMAND ----------

# renamed_df3 = spark.sql("select OBJECT_ID, DAYTIME, SD_TYPE, YEAR, MONTH, DAY, HOUR, MM, RUL, CYCLE, EVENT_ID from rc_data_06 where EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps>1440) order by YEAR, MONTH, DAY, HOUR, MM, RUL")


# COMMAND ----------

#10.1 - groupBy with count transactions before UCM event (minute time slot) by Event Id
renamed_df2.toPandas()

# COMMAND ----------

#10.1 - groupBy with count transactions before UCM event (minute time slot) by Event Id
renamed_df3.toPandas()

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
renamed_df5 = spark.sql(" select IN_RUL, "
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
                      + " from rc_data_06 where "
                      + " IN_RUL between -1440 and -1 " 
                      + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 "
                      + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 1440) "
                      + " group by IN_RUL order by IN_RUL "
                     )


# COMMAND ----------

(renamed_df5.columns)

# COMMAND ----------

renamed_df5.count()

# COMMAND ----------

columnList = [ 'RESI_LL_LUBE_OIL_PRESS',
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



pd_df_1 = (renamed_df5.toPandas())



# COMMAND ----------

pd_df_1.sort_values('IN_RUL',ascending=True)
##pd_df_1.describe()

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

def plot_rolling_average(xSortedList,ySortedList, w=15):
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
    plt.title('Rolling Mean & SD of '+ ySortedList.name, fontsize=22)
    plt.xlabel(xLabel)
    plt.ylabel("Sensor Value")
    plt.savefig(path+"img/"+timestamp+"_"+ySortedList.name+str(w)+"_moving_average.png")
    plt.show(block=False)
    return

def plot_rolling_average_4wSensors(x,y1,y2,y3,y4, w=15):
    
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
    plt.savefig(path+"img/"+timestamp+"_"+"rolling_windows.png")
    plt.show()

    return

def plot_rolling_average_4w(x,y, w=15):
    
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
    plt.savefig(path+'img/'+timestamp+'rolling_windows.png')
    plt.show()

    return


# COMMAND ----------

windowtime = 15

## Moviong Avg 15 time stpes....
plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],
                               pd_df_1['RESI_LL_LUBE_OIL_PRESS'],
                               pd_df_1['RESI_HH_LUBE_OIL_TEMP'],
                               pd_df_1['RESI_LL_SPEED'],
                               pd_df_1['RESI_HH_SPEED'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],
                               pd_df_1['RESI_LL_VIBRATION'],
                               pd_df_1['RESI_HH_VIBRATION'],
                               pd_df_1['RESI_HH_THROW1_DIS_TEMP'],
                               pd_df_1['RESI_HH_THROW1_SUC_TEMP'],windowtime)


plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['LUBE_OIL_PRESS'],
                               pd_df_1['LUBE_OIL_TEMP'],
                               pd_df_1['THROW_1_DISC_PRESS'],pd_df_1['THROW_1_DISC_TEMP'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['THROW_1_SUC_PRESS'],
                               pd_df_1['THROW_1_SUC_TEMP'],
                               pd_df_1['THROW_2_DISC_PRESS'],pd_df_1['THROW_2_DISC_TEMP'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['THROW_2_SUC_TEMP'],
                               pd_df_1['THROW_3_DISC_PRESS'],
                               pd_df_1['THROW_3_DISC_TEMP'],pd_df_1['THROW_3_SUC_PRESS'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['THROW_3_SUC_TEMP'],
                               pd_df_1['THROW_4_DISC_PRESS'],
                               pd_df_1['THROW_4_DISC_TEMP'],pd_df_1['VIBRATION'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['CYL_1_TEMP'],
                               pd_df_1['CYL_2_TEMP'],
                               pd_df_1['CYL_3_TEMP'],pd_df_1['CYL_4_TEMP'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['CYL_5_TEMP'],
                               pd_df_1['CYL_6_TEMP'],
                               pd_df_1['CYL_7_TEMP'],pd_df_1['CYL_8_TEMP'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['CYL_9_TEMP'],
                               pd_df_1['CYL_10_TEMP'],
                               pd_df_1['CYL_11_TEMP'],pd_df_1['FUEL_GAS_PRESS'],windowtime)

plot_rolling_average_4wSensors(pd_df_1['IN_RUL'],pd_df_1['LUBE_OIL_PRESS_ENGINE'],
                               pd_df_1['SPEED'],
                               pd_df_1['VIBRA_ENGINE'],pd_df_1['VIBRA_ENGINE'],windowtime)

# plot_rolling_average_4w(pd_df_1['IN_RUL'],pd_df_1['LUBE_OIL_PRESS_ENGINE'],windowtime)
# plot_rolling_average_4w(pd_df_1['IN_RUL'],pd_df_1['SPEED'],windowtime)
# plot_rolling_average_4w(pd_df_1['IN_RUL'],pd_df_1['VIBRA_ENGINE'],windowtime)




# COMMAND ----------

## Print Single Sensor Moving Average Per Invese RUL (Minute)
selectedCols = columnList 
for colName in selectedCols:
    #pd_df_1[colName].head()
    plot_rolling_average(pd_df_1['IN_RUL'], pd_df_1[colName],windowtime)


# COMMAND ----------

# MAGIC %md ## Seasonality
# MAGIC People tend to go on vacation mainly during summer holidays. That is, at some time periods during the year people tend to use aircrafts more frequently. We could check this hypothesis of a seasonal effect by 

# COMMAND ----------

df_piv_line = spark.sql(" select YEAR, MONTH , RUL, "
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
                      + " from rc_data_06 where "
                      + " RUL between 1 and 1440  " 
                      + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 "
                      + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > 1440 )"
                      + " group by YEAR, MONTH, RUL"
                      + " order by YEAR, MONTH, RUL")



# COMMAND ----------

df_piv_line.groupby("YEAR").count().toPandas()

# COMMAND ----------


pd_df = (df_piv_line.toPandas())


# COMMAND ----------

pd_df.describe()

# COMMAND ----------

pd_df.to_csv('/home/admin/data/rc/descibe_data_05_clean.csv')

# COMMAND ----------

# create line plot
def plot_colormap_season(pivot_df, parameterName):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    pivot_df.plot(colormap='jet');
    plt.title('Seasonal Effect per Month of '+parameterName, fontsize=24)
    plt.ylabel('Sensor Value')
    #plt.get_legend().remove()
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(path+'img/seasonal_effect_lines_400to1440_'+timestamp+"_"+parameterName+'.png')
    plt.show()
    sns.heatmap(pivot_df, annot=False)
    return

# COMMAND ----------

# create nice axes names
month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')
# reshape date

def plot_boxPlot(df_piv_box, parameterName):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    ##print(df_piv_box)
    # create a box plot
    fig, ax = plt.subplots()
    df_piv_box.plot(ax=ax, kind='box')
    ax.set_title('Seasonal Effect per Month', fontsize=24)
    ax.set_xlabel('Month')
    ax.set_ylabel(parameterName)
    ax.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.savefig(path+'img/seasonal_effect_boxplot_400to1440_'+timestamp+"_"+parameterName+'.png')
    plt.show()
    # plot heatmap
    sns.heatmap(df_piv_box, annot=False)
    return

# COMMAND ----------

## Print Single Sensor Moving Average Per Invese RUL (Minute)
selectedCols2 = columnList = [ 'RESI_LL_LUBE_OIL_PRESS',
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
    df_piv_box = pd_df.pivot_table(index=['RUL'], columns='MONTH',values= colName , aggfunc='mean')
    plot_boxPlot(df_piv_box,colName)
    
print("*** End **** ")

# COMMAND ----------

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
                      + " from rc_data_06 where "
                      + " RUL between 1 and 7200  " 
                      + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 "
                      + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps >= 7200 )"
                      + " group by IN_RUL , EVENT_ID "
                      + " order by IN_RUL , EVENT_ID ")



# COMMAND ----------

df_simple2.columns

# COMMAND ----------

#df_simple2.head()
pd_df_2 = (df_simple2.toPandas())

# COMMAND ----------

pd_df_2.columns

# COMMAND ----------

pd_df_2.describe()

# COMMAND ----------

# create line plot
def plot_line_graph(pivot_df, parameterName):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    pivot_df.plot(colormap='jet');
    plt.title(' Line Plot of Sensors : '+parameterName, fontsize=22)
    plt.ylabel('Sensor Value')
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(path+'img/seasonal_effect_lines_less_1to7200times_'+timestamp+"_"+parameterName+'.png')
    plt.show()
    return

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
    
for colName in selectedCols3:
    df_piv_box = pd_df_2.pivot_table(index=['IN_RUL'], columns='EVENT_ID',values= colName , aggfunc='mean')
    #df_piv_box
    plot_line_graph(df_piv_box,colName)
    

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

# statistical modeling libraries
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
#from arch import arch_model

# COMMAND ----------

sensor_x = 'LUBE_OIL_PRESS'

df_simple2 = spark.sql(" select  IN_RUL , MONTH ,"
                        +" RESI_LL_LUBE_OIL_PRESS AS RESI_LL_LUBE_OIL_PRESS,"
                        +" RESI_HH_LUBE_OIL_TEMP AS  RESI_HH_LUBE_OIL_TEMP,"
                        +" RESI_LL_SPEED AS  RESI_LL_SPEED,"
                        +" RESI_HH_SPEED AS  RESI_HH_SPEED,"
                        +" RESI_LL_VIBRATION AS  RESI_LL_VIBRATION,"
                        +" RESI_HH_VIBRATION AS  RESI_HH_VIBRATION,"
                        +" RESI_HH_THROW1_DIS_TEMP AS  RESI_HH_THROW1_DIS_TEMP,"
                        +" RESI_HH_THROW1_SUC_TEMP AS  RESI_HH_THROW1_SUC_TEMP,"
                        +" LUBE_OIL_PRESS AS  LUBE_OIL_PRESS,"
                        +" LUBE_OIL_TEMP AS  LUBE_OIL_TEMP,"
                        +" THROW_1_DISC_PRESS AS  THROW_1_DISC_PRESS,"
                        +" THROW_1_DISC_TEMP AS  THROW_1_DISC_TEMP,"
                        +" THROW_1_SUC_PRESS AS  THROW_1_SUC_PRESS,"
                        +" THROW_1_SUC_TEMP AS  THROW_1_SUC_TEMP,"
                        +" THROW_2_DISC_PRESS AS  THROW_2_DISC_PRESS,"
                        +" THROW_2_DISC_TEMP AS  THROW_2_DISC_TEMP,"
                        +" THROW_2_SUC_TEMP AS  THROW_2_SUC_TEMP,"
                        +" THROW_3_DISC_PRESS AS  THROW_3_DISC_PRESS,"
                        +" THROW_3_DISC_TEMP AS  THROW_3_DISC_TEMP,"
                        +" THROW_3_SUC_PRESS AS  THROW_3_SUC_PRESS,"
                        +" THROW_3_SUC_TEMP AS  THROW_3_SUC_TEMP,"
                        +" THROW_4_DISC_PRESS AS  THROW_4_DISC_PRESS,"
                        +" THROW_4_DISC_TEMP AS  THROW_4_DISC_TEMP,"
                        +" VIBRATION AS  VIBRATION,"
                        +" CYL_1_TEMP AS  CYL_1_TEMP,"
                        +" CYL_2_TEMP AS  CYL_2_TEMP,"
                        +" CYL_3_TEMP AS  CYL_3_TEMP,"
                        +" CYL_4_TEMP AS  CYL_4_TEMP,"
                        +" CYL_5_TEMP AS  CYL_5_TEMP,"
                        +" CYL_6_TEMP AS  CYL_6_TEMP,"
                        +" CYL_7_TEMP AS  CYL_7_TEMP,"
                        +" CYL_8_TEMP AS  CYL_8_TEMP,"
                        +" CYL_9_TEMP AS  CYL_9_TEMP,"
                        +" CYL_10_TEMP AS  CYL_10_TEMP,"
                        +" CYL_11_TEMP AS  CYL_11_TEMP,"
                        +" FUEL_GAS_PRESS AS  FUEL_GAS_PRESS,"
                        +" LUBE_OIL_PRESS_ENGINE AS  LUBE_OIL_PRESS_ENGINE,"
                        +" SPEED AS  SPEED,"
                        +" VIBRA_ENGINE AS  VIBRA_ENGINE "
                      + " from rc_data_06 where "
                      + " RUL between 1 and 100  "
                      + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 "
                      + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps >= 600 "
                      + " order by IN_RUL , MONTH ")





# COMMAND ----------


#AVG_LUBE_OIL_PRESS2 = df_spark_2['AVG_LUBE_OIL_PRESS']


pd_df2 = (df_simple2.toPandas())

#reshape date
df_piv_box = pd_df2.pivot_table(index=['IN_RUL'], columns='MONTH',values= sensor_x, aggfunc='mean')
print(df_piv_box)

# COMMAND ----------

pd_df2.index = pd_df2.IN_RUL
pd_df2.head(10)

import statsmodels.api as sm
decomp = sm.tsa.seasonal_decompose(pd_df2[0])
decomp.plot()
plt.show()

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