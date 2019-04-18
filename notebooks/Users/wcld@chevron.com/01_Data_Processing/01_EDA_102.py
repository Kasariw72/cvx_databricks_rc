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
#import pandas_datareader.data as web
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
#from ipywidgets import interactive, widgets, RadioButtons, ToggleButtons, Select, FloatSlider, FloatProgress
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
# sc = SparkContext()
# print(sc)

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir RCPredictiveMA/img/trend

# COMMAND ----------

# dbutils.fs.mv("dbfs:/FileStore/tables/tmp_rc/data07_20190126_1440T_CLEAN.csv", "dbfs:/RC/datasets")

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "/databricks/driver/RCPredictiveMA/"
sparkAppName = "RCDataAnlaysisTrend"
hdfsPathCSV = "dbfs:/RC/datasets/data07_20190126_1440T_CLEAN.csv"

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

### This step is for cleaning data using simple median value of each colum to the missing or unknown value of sensors.
from pyspark.sql.functions import when

def replaceByMedian(pySparkDF, columnList):
    for colName in columnList:
        med = pySparkDF.approxQuantile(colName, [0.5], 0.25)
        pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName]==8888888888,pySparkDF[colName]).otherwise(med[0]))
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

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data_05")
## Filter out the data during RC S/D
renamed_df = spark.sql("select * from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 0 ")

# COMMAND ----------

renamed_df3 = spark.sql("select EVENT_ID, CODE , COUNT(*) as num_steps from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")
renamed_df3.createOrReplaceTempView("tmp_step_event")
renamed_df3 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 GROUP BY  CODE, EVENT_ID")

# COMMAND ----------

import matplotlib.pyplot as plt
dataSet = renamed_df.toPandas()

fig, ax = plt.subplots()
ax.scatter(dataSet['SPEED'], dataSet['LABEL1']) #scatterplot
ax.plot(dataSet['SPEED', regressor.predict(dataSet['SPEED'), color = 'blue') #line
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")
plt.title("Plot title")
display(fig)

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

# %sh
# mv /databricks/driver/RCPredictiveMA/img/*.png /databricks/driver/RCPredictiveMA/img/trend

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trend
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

P_START_RUL = 420
P_END_RUL = 1820
P_MINIMUM_RUL = 420
P_CUR_SD_TYPE = "NMD"
windowtime = 15
selectedCols = columnList 

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

# MAGIC %sh
# MAGIC zip RCPredictiveMATrend.zip /databricks/driver/RCPredictiveMA/img/*.png

# COMMAND ----------

# MAGIC %sh
# MAGIC #cd RCPredictiveMA/img/
# MAGIC #mkdir tmp
# MAGIC #mv RCPredictiveMA/img/*.png tmp
# MAGIC # cd tmp
# MAGIC # ls -al

# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/RCPredictiveMA_MA3.zip", "dbfs:/mnt/Exploratory/WCLD/")

# COMMAND ----------

# MAGIC %md ## Seasonality
# MAGIC People tend to go on vacation mainly during summer holidays. That is, at some time periods during the year people tend to use aircrafts more frequently. We could check this hypothesis of a seasonal effect by 

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
                            +"AVG( FUEL_GAS_PRESS) AS  FUEL_GAS_PRESS,"
                            +"AVG( LUBE_OIL_PRESS_ENGINE) AS  LUBE_OIL_PRESS_ENGINE,"
                            +"AVG( SPEED) AS  SPEED,"
                            +"AVG( VIBRA_ENGINE) AS  VIBRA_ENGINE "
                          + " from rc_data_05 where "
                          + " RUL between "+ str(P_START_RUL) +" and "+str(P_END_RUL) 
                          + " and EVENT_ID <> 9999999 and CYCLE <> 9999999 and RUL<>0 "
                          + " and BE_SD_TYPE ='" + P_CUR_SD_TYPE+ "'"
                          + " and EVENT_ID IN (SELECT EVENT_ID FROM tmp_step_event where num_steps > "+str(P_MINIMUM_RUL)+" )"
                          + " group by IN_RUL ,MONTH, EVENT_ID "
                          + " order by IN_RUL ,MONTH, EVENT_ID ")
    return df_simple2


# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir /databricks/driver/RCPredictiveMA/img

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

#P_SD_TYPES = ["NMD","SCM","STB","SPM","UCM"]
P_SD_TYPES = ["UCM","SPM"]
P_START_RUL = 420
P_END_RUL = 1860
P_MINIMUM_RUL = 930
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

print(path)

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -al /databricks/driver/RCPredictiveMA/img

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /databricks/driver/RCPredictiveMA/img
# MAGIC ls -al

# COMMAND ----------

#timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# COMMAND ----------

# MAGIC %sh
# MAGIC zip img_result.zip *.png
# MAGIC ls -al
# MAGIC pwd

# COMMAND ----------

/databricks/driver/RCPredictiveMA/img_result.zip

# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/RCPredictiveMA/img/img_result.zip", "dbfs:/mnt/Exploratory/WCLD/")

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/RCPredictiveMA/img/img_result.zip", "dbfs:/RC/exploratory")

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/RC/exploratory")

# COMMAND ----------

# MAGIC %md 
# MAGIC <strong>Inference</strong>
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

# MAGIC %sh md
# MAGIC <strong>Inference</strong>
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

# COMMAND ----------

sample_img_dir = "file:/databricks/driver/RCPredictiveMA/img"
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

from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

destPath = "dbfs:/RC/datasets/output/img"


for imgPath in listImg:
  print(imgPath[0])
  dbutils.fs.cp(imgPath[0], destPath)
  img = mpimg.imread(imgPath[0])
  plt.imshow(img)
  display()
  print()

# COMMAND ----------

listImg = dbutils.fs.ls(destPath)

# COMMAND ----------

displayHTML('''<img src="dbfs:/RC/datasets/output/img/SPM_SEL_HEATMAP_20190129032416_RESI_LL_LUBE_OIL_PRESS_420_1860.png" style="width:600px;height:600px;">''')

# COMMAND ----------

for imgPath in listImg:
  print(imgPath[0])
  #displayHTML('''<img src="files/RC/SPM_SEL_HEATMAP_20190129032416_RESI_LL_LUBE_OIL_PRESS_420_1860.png" style="width:600px;height:600px;">''')
  #dbutils.fs.cp(imgPath[0], destPath)
  #img = mpimg.imread(imgPath[0])
  #plt.imshow(img)
  #display()
  
  print()

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install plotly

# COMMAND ----------

from plotly.offline import plot
from plotly.graph_objs import *
import numpy as np

x = np.random.randn(2000)
y = np.random.randn(2000)

# Instead of simply calling plot(...), store your plot as a variable and pass it to displayHTML().
# Make sure to specify output_type='div' as a keyword argument.
# (Note that if you call displayHTML() multiple times in the same cell, only the last will take effect.)

p = plot(
  [
    Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
    Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=3, opacity=0.3))
  ],
  output_type='div'
)

displayHTML(p)

# COMMAND ----------

import plotly 
plotly.tools.set_credentials_file(username='DemoAccount', api_key='lr1c37zw81')

# COMMAND ----------

import os
import urllib
model_dir = 'file:/databricks/driver/RCPredictiveMA/img/'

def display_image(url):
  """Downloads an image from a URL and displays it in Databricks."""
  filename = url.split('/')[-1]
  filepath = os.path.join(model_dir, filename)
  urllib.request.urlretrieve(url, filepath)
  image = os.path.join(model_dir, filename)
  image_png = image.replace('.jpg','.png')
  Image.open(image).save(image_png,'PNG')
  img = mpimg.imread(image_png)
  plt.imshow(img)
  display()
  
  
from pyspark.ml.image import ImageSchema
for imgPath in listImg:
  #print(imgPath[0])
  #image_df = ImageSchema.readImages(imgPath[0])
  display_image(imgPath[0])

# COMMAND ----------

dispaly("file:/databricks/driver/RCPredictiveMA/img/UCM_SEL_HEATMAP_20190128155807_THROW_4_DISC_TEMP_420_1860.png")

# COMMAND ----------

displayHTML("<img src='File:/databricks/driver/RCPredictiveMA/img/SPM_SEL_LINE_20190129032423_RESI_HH_THROW1_DIS_TEMP_420_1860.png' height='200' width='200'>")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

