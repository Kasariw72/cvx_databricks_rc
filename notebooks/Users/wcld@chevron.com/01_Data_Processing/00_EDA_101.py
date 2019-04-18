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
#from IPython.display import set_matplotlib_formats, Image

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

import datetime

# COMMAND ----------

#2 - Create SparkContext
# sc = SparkContext()
# print(sc)

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir RCPredictiveMA

# COMMAND ----------

# dbutils.fs.mv("dbfs:/FileStore/tables/tmp_rc/data07_20190126_1440T_CLEAN.csv", "dbfs:/RC/datasets")

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "/databricks/driver/RCPredictiveMA/"
sparkAppName = "RCDataAnlaysisTrend"
hdfsPathCSV = "dbfs:/RC/MSATER_DATA_FILES/data08_20190131_ALL_CLEAN.csv"
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
expContext = "00_eda_101_"+str(timestamp)+"_"

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

#Count sample of each quipment
display(renamed_df.groupBy("BE_SD_TYPE").count())

# COMMAND ----------

## Coount number of sample per year
display(renamed_df.where("BE_SD_TYPE='UCM'").groupBy("YEAR").count())

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

#12 - compute dataframe using sql command via string
renamed_df.createOrReplaceTempView("rc_data_05")
## Filter out the data during RC S/D
renamed_df = spark.sql("select * from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 0 ")

# COMMAND ----------

display(renamed_df.groupBy("BE_SD_TYPE").count())

# COMMAND ----------

# datafile = expContext+"_RC_MED_DATA.csv"
# ##pdDF = renamed_df.where("BE_SD_TYPE='SPM'").toPandas()

# pdDF.to_csv('File:'+path+'/'+datafile)

# train_df.to_csv(datafile, encoding='utf-8',index = None)
# dbutils.fs.mv(dirPath+datafile, dataLake)

# COMMAND ----------

renamed_df3 = spark.sql("select EVENT_ID, CODE ,YEAR, COUNT(*) as num_steps from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 and BE_SD_TYPE='UCM' GROUP BY  CODE, EVENT_ID,YEAR")
renamed_df3.createOrReplaceTempView("tmp_step_event")
renamed_df3 = spark.sql("select EVENT_ID, CODE, YEAR, COUNT(*) as num_steps from rc_data_05 where EVENT_ID <> 9999999 and RUL <> 9999999 and RUL <> 0 and BE_SD_TYPE='UCM' GROUP BY  CODE, EVENT_ID,YEAR")

# COMMAND ----------

# MAGIC %md #Count UCM Events per Equipment per Year

# COMMAND ----------

display(renamed_df3.toPandas())

# COMMAND ----------

pdSpark = spark.sql("SELECT YEAR, BE_SD_TYPE, CODE, COUNT(DISTINCT EVENT_ID) AS EVENT_NUM FROM rc_data_05 GROUP BY YEAR, BE_SD_TYPE, CODE ORDER BY YEAR, BE_SD_TYPE, CODE ")


# COMMAND ----------

display(pdSpark.toPandas())

# COMMAND ----------

# MAGIC %md 
# MAGIC Rank the most long running or equipment life time (Counting from start to stop = 1 EVENT)

# COMMAND ----------

historyDF = spark.sql(" SELECT * FROM tmp_step_event where num_steps between 1440 and 2880 and EVENT_ID in (select EVENT_ID from rc_data_05 where BE_SD_TYPE = 'UCM' ) ORDER BY num_steps asc")
display(historyDF.toPandas())

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from data08 where EVENT_ID in (1787,1334,2878) and N_MM in (1,15,30,45,59)

# COMMAND ----------

# MAGIC %md ## Seasonality

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

print(expContext)

# COMMAND ----------

# MAGIC %sh
# MAGIC #mkdir /databricks/driver/RCPredictiveMA/img
# MAGIC #mkdir /databricks/driver/RCPredictiveMA/img/00_eda_101_201902010308

# COMMAND ----------

path = "/databricks/driver/RCPredictiveMA/img/00_eda_101_201902010308/"

# COMMAND ----------

# create line plot
def plot_line_graph(pivot_df, parameterName,sd_type,rul_start,rul_end):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    imgPath = path+sd_type+'_SEL_LINE_'+timestamp+"_"+parameterName+"_"+str(rul_start)+"_"+str(rul_end)+'.png'
    
    pivot_df.plot(colormap='jet');
    plt.title("Line Plot of Sensor: "+parameterName+"("+sd_type+")"+str(rul_start)+" to "+str(rul_end), fontsize=18)
    plt.ylabel('Sensor Value')
    plt.legend(loc='best', bbox_to_anchor=(1.0, 0.5))
    plt.savefig(imgPath)
    ##plt.show()
    ##display(plt)
    return

# COMMAND ----------

# # create nice axes names
# month_names = pd.date_range(start='1949-01-01', periods=12, freq='MS').strftime('%b')
# # reshape date
def plot_boxPlot(pivot_df, parameterName,sd_type,rul_start,rul_end):
    ##print(df_piv_box)
    # create a box plot
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    imgPathBoxPlot = path+sd_type+'_SEL_BOX_'+timestamp+"_"+parameterName+"_"+str(rul_start)+"_"+str(rul_end)+'.png'
    imgPathHeat =path+sd_type+'_SEL_HEATMAP_'+timestamp+"_"+parameterName+"_"+str(rul_start)+"_"+str(rul_end)+'.png'
    
    fig, ax = plt.subplots()
    pivot_df.plot(ax=ax, kind='box')
    ax.set_title('Seasonal Effect per Month', fontsize=24)
    ax.set_xlabel('Month')
    ax.set_ylabel(parameterName)
    ax.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    plt.savefig(imgPathBoxPlot)
    ##dispay(fig)
    ##plt.show()
    ##plot heatmap
    sns_heatmap = sns.heatmap(pivot_df, annot=False)
    display(fig)
    fig = sns_heatmap.get_figure()
    fig.savefig(imgPathHeat)
    return

# COMMAND ----------

def generateLineGraph(P_SD_TYPES,P_START_RUL,P_END_RUL,P_MINIMUM_RUL):
  for P_CUR_SD_TYPE in P_SD_TYPES:

      spDF = queryAVGSensorByEventId(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL)
      print("** Start drawing graph of "+P_CUR_SD_TYPE)
      
      pd_df_1 = (spDF.toPandas())
      for colName in selectedCols:
          
          df_piv_box = pd_df_1.pivot_table(index=['IN_RUL'], columns='EVENT_ID',values= colName , aggfunc='mean')
          plot_line_graph(df_piv_box,colName,P_CUR_SD_TYPE,P_START_RUL,P_END_RUL)
          print("** Saved line graph of "+colName)
          
      print("** End drawing graph of "+P_CUR_SD_TYPE)
  print("**Finish all drawing graph**")
  return

def generateBoxHeatGraph(P_SD_TYPES,P_START_RUL,P_END_RUL,P_MINIMUM_RUL):
  
  for P_CUR_SD_TYPE in P_SD_TYPES:
      spDF = queryAVGSensorByEventId(P_CUR_SD_TYPE,P_START_RUL,P_END_RUL,P_MINIMUM_RUL)
      print("** Start drawing graph of "+P_CUR_SD_TYPE)
      pd_df_1 = (spDF.toPandas())
      
      for colName in selectedCols:
        
          df_piv_box = pd_df_1.pivot_table(index=['MONTH'], columns='IN_RUL',values= colName , aggfunc='mean')
          plot_boxPlot(df_piv_box,colName,P_CUR_SD_TYPE,P_START_RUL,P_END_RUL)
          print("** Saved box and heatmap graphs of "+colName)
          
      print("** End drawing graph of "+P_CUR_SD_TYPE)
  print("**Finish all drawing graph**")
  return

# COMMAND ----------

selectedCols = ['RESI_LL_LUBE_OIL_PRESS',
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

P_SD_TYPES = ["UCM","NMD"]
#P_SD_TYPES = ["UCM","SPM"]
RUL_SETs = [[1,720],[1,1440],[1,2880],[1,1860],[1,4320]]
P_START_RUL = 1
P_END_RUL   = 1440
P_MINIMUM_RUL = 420
P_CUR_SD_TYPE = "UCM"

# for RULStartEnd in RUL_SETs:
#   print("RUL between ",RULStartEnd[0]," and ",RULStartEnd[1])
#   P_START_RUL = RULStartEnd[0]
#   P_END_RUL = RULStartEnd[1]
#   generateLineGraph(P_SD_TYPES,P_START_RUL,P_END_RUL,P_MINIMUM_RUL)
#   generateBoxHeatGraph(P_SD_TYPES,P_START_RUL,P_END_RUL,P_MINIMUM_RUL)

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir RCPredictiveMA/RC_EVENTS

# COMMAND ----------

# pdSpark.write.format("csv").save(pathDataFileDir)

# COMMAND ----------

# dbutils.fs.mv("dbfs:/databricks/driver/RCPredictiveMA/datafiles")

# COMMAND ----------

# %sh
# rm databricks/driver/RCPredictiveMA/datafiles/*.*

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/mnt/Exploratory/WCLD/BetaProject/RAWDATAFILES")

# COMMAND ----------


# renamed_df3.toPandas().sort_values("num_steps", ascending=False)
def exportCSV(p_sd_type,p_rows,path):
    renamed_df3 = spark.sql("select EVENT_ID, CODE, COUNT(*) as num_steps from rc_data_05 where BE_SD_TYPE ='"+p_sd_type+"' and EVENT_ID <> 9999999 and RUL <> 9999999 GROUP BY  CODE, EVENT_ID")
    
    events = renamed_df3.toPandas()
    
    renamed_df3 = spark.sql("select EVENT_ID, CODE,  num_steps from tmp_step_event where  num_steps>="+str(p_rows)+" order by num_steps desc")
    
    for event_id in events.EVENT_ID:
        
        pathCSV = path
        
        tdf = renamed_df.where(" RUL <>0 AND EVENT_ID = "+str(event_id))

        numrows = tdf.count()
        rcCode = tdf.first().CODE[:4]
        
        filePath = "file:/databricks/driver/"+pathCSV+rcCode+"_ID_"+str(event_id)+"_"+str(numrows)+".csv"
        tdf.toPandas().to_csv(filePath)
        print(event_id, ":", rcCode , " >> ", numrows)
        
    print("****** End Exporting Data Files at ", pathCSV)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al RCPredictiveMA/RC_EVENTS/BEWK_ID_2580_1859.csv

# COMMAND ----------

#dbutils.fs.cp("file:/databricks/driver/RCPredictiveMA/RC_EVENTS/BEWK_ID_2580_1859.csv", "dbfs:/RC/datafiles")
dbutils.fs.cp("dbfs:/mnt/Exploratory/WCLD/BetaProject/data_files.zip", "file:/databricks/driver/RCPredictiveMA/")

#import shutil
#shutil.mv(file:/databricks/driver/RCPredictiveMA/RC_EVENTS/, "/dbfs/tensorflow/logs")

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /databricks/driver/RCPredictiveMA/data_files.zip

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/RC/RC_EVENTS")

# COMMAND ----------

#dbutils.fs.rm("dbfs:/RC/SPM")
%fs 
dbfs -rm RC/datasets/data07_20190126_1440T_CLEAN.csv

# COMMAND ----------

import shutil
shutil.move("RCPredictiveMA/RC_EVENTS", "dbfs:/RC/RC_EVENTS")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al databricks/driver/RCPredictiveMA/RC_EVENTS/

# COMMAND ----------

pathDataFileDir = "RCPredictiveMA/RC_EVENTS/"

for sdType in P_SD_TYPES:
  exportCSV(sdType,P_MINIMUM_RUL,pathDataFileDir)

# COMMAND ----------

# MAGIC %sh
# MAGIC pwd

# COMMAND ----------

print(pathDataFileDir)

# COMMAND ----------

# MAGIC %sh
# MAGIC zip data_files.zip /databricks/driver/RCPredictiveMA/RC_EVENTS/*.csv

# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/data_files.zip", "dbfs:/mnt/Exploratory/WCLD/BetaProject")

# COMMAND ----------

# %sh 
# ls -al /databricks/driver/RCPredictiveMA/

# COMMAND ----------

# %sh
# cd /databricks/driver/RCPredictiveMA/img
# ls -al


# COMMAND ----------

# %sh
# zip RCPredictiveMA_result2.zip /databricks/driver/RCPredictiveMA/img/*.png

# COMMAND ----------

# %sh
# ls -al
# # pwd

# COMMAND ----------

# dbutils.fs.cp("file:/databricks/driver/RCPredictiveMA_result2.zip", "dbfs:/mnt/Exploratory/WCLD/BetaProject")

# COMMAND ----------

# dbutils.fs.mv("file:/databricks/driver/RCPredictiveMA/img/img_result.zip", "dbfs:/RC/exploratory")

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

