# Databricks notebook source
# MAGIC %md ### 
# MAGIC # Bidirectional LSTM
# MAGIC # Binary-Multiple Classification
# MAGIC # Attention Mechanism
# MAGIC Predict if an asset will fail within certain time frame (e.g. cycles)

# COMMAND ----------

from keras import backend as K
from keras.utils import multi_gpu_model
from keras.utils import plot_model

from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import *

# Horovod: Import the relevant submodule
import horovod.keras as hvd
from sparkdl import HorovodRunner

import keras
import tensorflow as tf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Setting seed for reproducibility
np.random.seed(1234)  
PYTHONHASHSEED = 0
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM,Flatten
from keras.callbacks import TensorBoard
# remove warnings
import warnings
warnings.filterwarnings('ignore')
# plotting and plot stying
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
#jupyter wdgets
#from ipywidgets import interactive, widgets, RadioButtons, ToggleButtons, Select, FloatSlider, FloatProgress
from IPython.display import set_matplotlib_formats, Image

# select a random sample without replacement
from random import seed
from random import sample
import math

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
#from attention_decoder import AttentionDecoder
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent
#, _time_distributed_dense
from keras.engine import InputSpec

from sklearn.metrics import classification_report

from keras.metrics import categorical_accuracy as metrics
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import sys
import traceback
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_fscore_support as score
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight

from sklearn.metrics import roc_auc_score

# COMMAND ----------

#1 - import module
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType

# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
tfb_log_dir ="/tmp/tfb_log_dir/"
sparkAppName = "16_BDLSTM_RV1"
# define path to save model
model_path = sparkAppName+".h5"

dirPath = "file:/databricks/driver/"
dataLake = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
mntDatalake = "/mnt/Exploratory/WCLD/"+sparkAppName

tf_at_dl_dir = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
dbFSDir = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV8"
dbfsDeltaDir = "dbfs:/RC/delta/events"
  
tensorboardLogPath = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName+"/tensorboard_log"

timestamp = datetime.datetime.now().strftime("%Y%m%d%H")

expContext = sparkAppName+"_"+str(timestamp)
tfb_log_dir = tfb_log_dir+expContext
print(expContext)

### Enable Tensorboard and save to the localtion
dbutils.tensorboard.start(tfb_log_dir)

# COMMAND ----------

selectedCols = ['CODE','YEAR','CYCLE', 'RUL','LABEL1','LABEL2','NOM_CYCLE','EVENT_ID','RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_LUBE_OIL_PRESS',
 'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',  'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP', 'LUBE_OIL_PRESS',  'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP', 'THROW_1_SUC_PRESS', 'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS', 'THROW_2_DISC_TEMP', 'THROW_2_SUC_PRESS', 'THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS', 'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP',
'CYL_6_TEMP', 'CYL_7_TEMP', 'CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP', 'CYL_12_TEMP', 'LUBE_OIL_PRESS_ENGINE', 'MANI_PRESS', 'RIGHT_BANK_EXH_TEMP', 'SPEED',
'VIBRA_ENGINE', 'S1_RECY_VALVE', 'S1_SUCT_PRESS', 'S1_SUCT_TEMPE', 'S2_STAGE_DISC_PRESS', 'S2_SCRU_SCRUB_LEVEL', 'GAS_LIFT_HEAD_PRESS', 'IN_CONT_CONT_VALVE', 'IN_SEP_PRESS']

sensor_cols = ['RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_LUBE_OIL_PRESS',
 'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',  'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP', 'LUBE_OIL_PRESS',  'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP', 'THROW_1_SUC_PRESS', 'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS', 'THROW_2_DISC_TEMP', 'THROW_2_SUC_PRESS', 'THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS', 'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP',
'CYL_6_TEMP', 'CYL_7_TEMP', 'CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP', 'CYL_12_TEMP', 'LUBE_OIL_PRESS_ENGINE', 'MANI_PRESS', 'RIGHT_BANK_EXH_TEMP', 'SPEED',
'VIBRA_ENGINE', 'S1_RECY_VALVE', 'S1_SUCT_PRESS', 'S1_SUCT_TEMPE', 'S2_STAGE_DISC_PRESS', 'S2_SCRU_SCRUB_LEVEL', 'GAS_LIFT_HEAD_PRESS', 'IN_CONT_CONT_VALVE', 'IN_SEP_PRESS']

# propertyCols = ['CODE','YEAR','CYCLE', 'RUL','LABEL1','LABEL2','BF_SD_TYPE','NOM_CYCLE']
propertyCols = ['CODE','YEAR','CYCLE', 'RUL','LABEL1','LABEL2','NOM_CYCLE']

intCols = ['YEAR','CYCLE', 'RUL','LABEL1','LABEL2','EVENT_ID']

propertyCol0 = ['(EVENT_ID + 10000000) as EVENT_ID']
propertyCol2 = ['(EVENT_ID + 20000000) as EVENT_ID']
propertyCol1 = ['(EVENT_ID + 30000000) as EVENT_ID']


dupColLabel0 = propertyCols + propertyCol0 + sensor_cols
dupColLabel2 = propertyCols + propertyCol2 + sensor_cols
dupColLabel1 = propertyCols + propertyCol1 + sensor_cols

w1 = 1440
w0 = 720

# pick a large window size of 60 cycles
# Target 1 days before failure 60 time steps
sequence_length = 100
input_length = 100

sensor_av_cols = ["AVG_" + nm for nm in sensor_cols]
sensor_sd_cols = ["SD_" + nm for nm in sensor_cols]

sequence_cols = sensor_cols + sensor_av_cols + sensor_sd_cols
#sequence_cols = sensor_cols
# print(sequence_cols)
columns_to_drop = ['CODE','YEAR']
#sequence_cols = sensor_cols

# COMMAND ----------

print(len(sequence_cols))

# COMMAND ----------

# MAGIC %md ## Data Ingestion
# MAGIC Training Data

# COMMAND ----------

def copyData(fromPath, toPath,recurseFlag=True):
  dbutils.fs.cp(fromPath, toPath,recurse=recurseFlag)
  print("Completed copying ["+fromPath+"] to "+toPath)
  
def round_down_float(n, decimals=0):
    multiplier = 10 ** decimals
    return float(math.floor(n * multiplier) / multiplier)

# COMMAND ----------

def normalizeMaxMinTrain(stdScaledDF,min_max_scaler):
 
  #train_df['cycle_norm'] = train_df['cycle']
  cols_normalize = stdScaledDF.columns.difference(['EVENT_ID','YEAR','CYCLE','RUL','LABEL1','LABEL2','NOM_CYCLE'])
  min_max_scaler = preprocessing.MinMaxScaler()
  norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(stdScaledDF[cols_normalize]), 
                               columns=cols_normalize, 
                               index=stdScaledDF.index)
  join_df = stdScaledDF[stdScaledDF.columns.difference(cols_normalize)].join(norm_train_df)
  stdScaledDF = join_df.reindex(columns = stdScaledDF.columns)
  print("Finish normalization train dataset!")
  return stdScaledDF, min_max_scaler


# COMMAND ----------

#3 - Setup SparkSession(SparkSQL)
spark = (SparkSession.builder.appName(sparkAppName).getOrCreate())
print(spark)
#4 - Read file to spark DataFrame
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark)

# COMMAND ----------

# events = spark.read.option("header","true").option("inferSchema", "true").csv("dbfs:/RC/RC_EVENTS/ALL_FILES_RV6")
# events.write.format("parquet").mode("overwrite").partitionBy("CODE").save("dbfs:/RC/events_parquet")
# display(spark.sql("DROP TABLE  IF EXISTS events"))
# events.write.format("delta").mode("overwrite").partitionBy("CODE").save("dbfs:/RC/delta/events")
# display(spark.sql("OPTIMIZE events ZORDER BY (EVENT_ID)"))

# COMMAND ----------

dfDirDBFS = dbutils.fs.ls(dbFSDir)
display(spark.sql("OPTIMIZE events ZORDER BY (EVENT_ID)"))
#spark.conf.set("spark.databricks.io.cache.enabled", "true")
df_delta = spark.read.format("delta").option("header","true").option("inferSchema", "true").load(dbfsDeltaDir)
df_delta = df_delta.selectExpr(selectedCols)
df_delta = convertStr2Int(df_delta, intCols, sensor_cols)

# COMMAND ----------

# MAGIC %md #Mini Batch Spark Data Loading

# COMMAND ----------

def upsampling(selectColDF,isTrain,isBinary):
  if isTrain:
    if isBinary:
      
      newUpDF0 = selectColDF.where("RUL BETWEEN 1 AND 1440")
      newUpDF0 = newUpDF0.selectExpr(dupColLabel0)
      
#       newUpDF1 = selectColDF.where("RUL BETWEEN 1 AND 1440")
#       newUpDF1 = newUpDF1.selectExpr(dupColLabel1)
      
#       newUpDF2 = selectColDF.where("RUL BETWEEN 1 AND 1440")
#       newUpDF2 = newUpDF2.selectExpr(dupColLabel1)
      
#      newDF1 = newUpDF0.union(newUpDF1)
#       newDF2 = newDF1.union(newUpDF2)
      return newUpDF0
    else:
      newUpDF2 = selectColDF.where("RUL BETWEEN 1 AND 720")
      newUpDF2 = newUpDF2.selectExpr(dupColLabel2)
      #Upsampling Positive Class Label 1
      newUpDF1 = selectColDF.where("RUL BETWEEN 721 AND 1440")
      newUpDF1 = newUpDF1.selectExpr(dupColLabel1)
    return newUpDF2.union(newUpDF1)
  else:
    if isBinary:
      #Upsampling Positive Class Label 1
      newUpDF1 = selectColDF.where("RUL BETWEEN 1441 AND 2881")
      return newUpDF1.selectExpr(dupColLabel1)
    else:
#       #Upsampling Positive Class Label 1
#       newUpDF1 = selectColDF.where("RUL BETWEEN 721 AND 1440")
#       newUpDF1 = newUpDF1.selectExpr(dupColLabel1)
#       #Upsampling Negative Class Label 0
      newUpDF0 = selectColDF.where("RUL BETWEEN 1441 AND 1800")
      newUpDF0 = newUpDF0.selectExpr(dupColLabel0)
      #return newUpDF1.union(newUpDF0)
      return newUpDF0

# COMMAND ----------

def convertStr2Int(pySparkDF, intColumns,doubleColumns):
    for colName in intColumns:
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(IntegerType()))    
      pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(0))
      
    for colName in doubleColumns:
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(DoubleType()))
      pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(0))
    return pySparkDF

def getFilePathInDBFS(dbFSDir,eventId):
  row = []
  for path in dfDirDBFS:
    row = path
    if row[0].__contains__("ID_"+str(eventId)):
      return row[0]
  return "DATA FILE NOT FOUND"

def getFilePaths(dbFSDir, eventIds):
  dicEventDataPath = {}
  for eventId in eventIds:
    eventFilePath = getFilePathInDBFS(dbFSDir, eventId)
    if eventFilePath !="DATA FILE NOT FOUND":
      dicEventDataPath[eventId]=eventFilePath
  
  if dicEventDataPath == {}:
    return "DATA FILE NOT FOUND"
  
  return dicEventDataPath

def getDataFromCSV(sqlContext, dbFSDir,eventIds, selectedCols, isTrainSet,isBinary=True):
  # read training data - It is the RC engine run-to-failure data.
  #df = spark.read.format("csv").option("inferSchema", "true").option("header","true").load(dataPath)
  if len(eventIds)<1:
    return "REQUIRE AT LEAST ONE EVENT_ID"
  else:
    print(eventIds)
    
  ##Test System
  samplePathDic = getFilePaths(dbFSDir,eventIds)
  
  #print(samplePathDic)
  
  if samplePathDic == "DATA FILE NOT FOUND":
    return "DATA FILE NOT FOUND"
  
  df = ""
  
  for eventId in eventIds:
    if eventId in samplePathDic:
      
      #print(">> Loading data from "+str(samplePathDic[eventId]))
      loadDF = (sqlContext.read.option("header","true").option("inferSchema", "true").csv(samplePathDic[eventId]))
      #loadDF = df_delta.where("EVENT_ID="+str(eventId))
      loadDF = loadDF.selectExpr(selectedCols)
      
      if df != "":
        df = df.union(loadDF)
      else:
        df = loadDF
      ##print("Loaded : ",eventId)
  
  ##df.cache()
  #print("Spark finishs caching dataframe!")
  selectColDF = df.selectExpr(selectedCols)
  selectColDF = convertStr2Int(selectColDF, intCols, sensor_cols)
  
  #selectColDF2 = df.selectExpr(selectedColUpSampling)
  #selectColDF2 = convertStr2Int(selectColDF2, intCols, sensor_cols)
  
  #compute dataframe using sql command via string
  selectColDF.createOrReplaceTempView("RC_DATA_TMP")
  #filter out the data during RC S/D
  selectColDF = spark.sql("SELECT * FROM RC_DATA_TMP WHERE RUL BETWEEN 1 AND 3600 ORDER BY YEAR, EVENT_ID, CYCLE")
  selectColDF = selectColDF.dropDuplicates()
  ## Up positive samples X2
  
  upsamplingDF = upsampling(selectColDF,isTrainSet,isBinary)
  resultDF = selectColDF.union(upsamplingDF)
  resultDF = resultDF.dropDuplicates(['EVENT_ID','CYCLE','RUL'])
  
#resultPD = resultDF.toPandas()
#resultDF = selectColDF
##selectDataCols = selectDataCols.sort_values(['YEAR','MONTH','DAY','EVENT_ID','CYCLE'])
  return resultDF.toPandas()

# COMMAND ----------

## Loading Index Data Table
indexFilePath = "dbfs:/RC/MSATER_DATA_FILES/EVENT_ID_DATA_SETS_FULL_V10.csv"
df2 = (sqlContext.read.option("header","true").option("inferSchema", "true").csv(indexFilePath))
# If the path don't have file:/// -> it will call hdfs instead of local file system
df2.cache()
print("Spark finishs caching:"+indexFilePath)

# COMMAND ----------

#compute dataframe using sql command via string
df2.createOrReplaceTempView("INDEX_RC_DATA")
sparkDF = spark.sql("SELECT * FROM INDEX_RC_DATA")

# COMMAND ----------

display(sparkDF.where("SET_TYPE<>'NA'"))

# COMMAND ----------

def add_features(df_in, rolling_win_size,sensor_cols):
    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.
    Args:
            df_in (dataframe)     : The input dataframe to be proccessed (training or test) 
            rolling_win_size (int): The window size, number of cycles for applying the rolling function
    Reurns:
            dataframe: contains the input dataframe with additional rolling mean and std for each sensor
    """
    
    #sensor_cols = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
    
    sensor_av_cols = ["AVG_" + nm for nm in sensor_cols]
    sensor_sd_cols = ["SD_" + nm for nm in sensor_cols]
    
#     print(sensor_av_cols)
#     print(sensor_sd_cols)
    
    df_out = pd.DataFrame()
    
    ws = rolling_win_size
    
    #calculate rolling stats for each engine id
    
    for m_id in pd.unique(df_in.EVENT_ID):
    
        # get a subset for each engine sensors
        df_engine = df_in[df_in['EVENT_ID'] == m_id]
        df_sub = df_engine[sensor_cols]
    
        # get rolling mean for the subset
        av = df_sub.rolling(ws, min_periods=1).mean()
        av.columns = sensor_av_cols
        
        # get the rolling standard deviation for the subset
        sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)
        sd.columns = sensor_sd_cols
        
        # combine the two new subset dataframes columns to the engine subset
        new_ftrs = pd.concat([df_engine,av,sd], axis=1)
    
        # add the new features rows to the output dataframe
        df_out = pd.concat([df_out,new_ftrs])
        
    return df_out

# COMMAND ----------

def loadDataSet(datasetType):
  totalEvents = 0
  dataEventMap = {}
  print("Loaded Training Sets")
  trainNumSetNo = sparkDF.select("NO_SET").where("SET_TYPE='"+datasetType+"'").distinct()
  if trainNumSetNo.count()>0:
    print("Found UCM Training Sets :",str(trainNumSetNo.count()))
    
    dataUCM = sparkDF.select("NO_SET").where("SET_TYPE='"+datasetType+"' and BF_SD_TYPE='UCM'").distinct().count()
    dataNMD = sparkDF.select("NO_SET").where("SET_TYPE='"+datasetType+"' and BF_SD_TYPE='NMD'").distinct().count()
    print(datasetType + " Sets : ",str(dataUCM+dataNMD),">> UCM/NMD:",str(dataUCM),"/",str(dataNMD))
    
    tSetIds = trainNumSetNo.toPandas()
    tSetIds = tSetIds.sort_values('NO_SET',ascending=True)
    
    for tId in tSetIds['NO_SET']:
      #print("tId: ",tId)
      eventIds = (sparkDF.select("EVENT_ID").where("SET_TYPE='"+datasetType+"' and NO_SET="+str(tId)).orderBy(["YEAR","MONTH","EVENT_ID"],ascending=[1,1,1])).toPandas()
      eventList = []
      index = 0
      for eId in eventIds['EVENT_ID']:
        #print("Inserted Event Id to List : ",str(eId))
        eventList.insert(index,eId)
        index = index + 1
        totalEvents = totalEvents + 1
      dataEventMap[tId]=eventList
  else:
    print("Error No UCM Training datasets in file : "+indexFilePath)
    
  return dataEventMap,totalEvents

# COMMAND ----------

# MAGIC %md ## Deep Learining - LSTM for Binary Classification Objective

# COMMAND ----------

# print(sensor_cols)

# COMMAND ----------

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

        # function to generate labels
def gen_labels(id_df, seq_length, label):
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]] 
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target. 
    return data_matrix[seq_length:num_elements, :]
  
def gen_data_train_val(targetColName, train_df,sequence_length, sequence_cols):
  
  # pick the feature columns 
  # generator for the sequences
  seq_gen = (list(gen_sequence(train_df[train_df['EVENT_ID']==id], sequence_length, sequence_cols)) 
             for id in train_df['EVENT_ID'].unique())

  # generate sequences and convert to numpy array
  seq_array = np.concatenate(list(seq_gen)).astype(np.float64)
  seq_array.shape
  
  # generate labels
  label_gen = [gen_labels(train_df[train_df['EVENT_ID']==id], sequence_length, [targetColName]) 
             for id in train_df['EVENT_ID'].unique()]
  label_array = np.concatenate(label_gen).astype(np.float64)
  label_array.shape

  nb_features = seq_array.shape[2]
  nb_out = label_array.shape[1]
  
  return seq_array, label_array, nb_features, nb_out

# COMMAND ----------

import os
import time

##WARNING: This example is using DBFS FUSE, which is not suitable for large scale distributed DL workloads! Setup DL storage and update this path.
FUSE_MOUNT_LOCATION = "/dbfs/mnt/Exploratory/WCLD/"+sparkAppName+"/model/hvd_checkpoint"

"dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
checkpoint_dir = FUSE_MOUNT_LOCATION + '{}/'.format(time.time())
os.makedirs(checkpoint_dir)

# COMMAND ----------

#Training Sets
totalEvents = 0
trainMap,totalEvents = loadDataSet("TRAIN")
print("Total Events in Traing Dataset : ",str(totalEvents))
#trainMap = {1: [679, 693, 694, 704, 705]}

# # trainMap = {1: [2091, 2101, 2103, 2104, 2119, 538, 546, 574, 578, 580, 660, 672],
# #             2: [679, 693, 694, 704, 705, 707, 712, 714, 732, 780, 787, 791, 815],
# #             3: [819, 842, 849, 852, 857, 860, 863, 2734, 2753, 3060, 3093, 3129], 
# #             4: [3134, 3533, 3559, 3755, 3772, 3774, 3781, 3793, 3812, 3826, 3837],
# #             5: [4373, 4445, 4449, 4641, 4772, 4851, 4955, 4969, 2771, 2789, 2853], 
# #             6: [3140, 3164, 3206, 3271, 3453, 3631, 3658, 3665, 3801, 3824, 3830]}

print(trainMap)
#Validation Sets
valMap , totalEvents = loadDataSet("VALI")
print("Total Events in Validation Dataset : ",str(totalEvents))
#valMap = {1: [3837]}
#valMap = {1: [679, 693, 694, 704, 705]}
print(valMap)

# COMMAND ----------

def printDFPortion(train_df, val_df, label):
  
  if label=="LABEL2":
    try:
      print("+++++Train Data Set LABEL2 ++++++++++++\n")
      print(train_df['LABEL2'].value_counts())
      print('\nClass 0 samples = {0:.0%}'.format(train_df['LABEL2'].value_counts()[0]/train_df['LABEL2'].count()))
      print('\nClass 1 samples = {0:.0%}'.format(train_df['LABEL2'].value_counts()[1]/train_df['LABEL2'].count()))
      print('\nClass 2 samples = {0:.0%}'.format(train_df['LABEL2'].value_counts()[2]/train_df['LABEL2'].count()))
    except:
      pass
    print("+++++Validation Data Set LABEL2 ++++++++++++\n")
    try:
      print(val_df['LABEL2'].value_counts()) 
      print('\nClass 0 samples = {0:.0%}'.format(val_df['LABEL2'].value_counts()[0]/val_df['LABEL2'].count()))
      print('\nClass 1 samples = {0:.0%}'.format(val_df['LABEL2'].value_counts()[1]/val_df['LABEL2'].count()))
      print('\nClass 2 samples = {0:.0%}'.format(val_df['LABEL2'].value_counts()[2]/val_df['LABEL2'].count()))
    except:
      pass
  else:
    try:
      print("+++++Train Data Set LABEL1 ++++++++++++\n")
      # print stat for binary classification label
      print(train_df['LABEL1'].value_counts())
      print('\nNegaitve samples [0] = {0:.0%}'.format(train_df['LABEL1'].value_counts()[0]/train_df['LABEL1'].count()))
      print('\nPosiitve samples [1] = {0:.0%}'.format(train_df['LABEL1'].value_counts()[1]/train_df['LABEL1'].count()))
    except:
      pass

    try:
      print("+++++Validation Data Set LABEL1 ++++++++++++\n")
      print(val_df['LABEL1'].value_counts())
      print('\nNegaitve samples [0] = {0:.0%}'.format(val_df['LABEL1'].value_counts()[0]/val_df['LABEL1'].count()))
      print('\nPosiitve samples [1] = {0:.0%}'.format(val_df['LABEL1'].value_counts()[1]/val_df['LABEL1'].count()))
    except:
      pass

# COMMAND ----------

def gen_data_test_val(targetColName, data_df,sequence_length, sequence_cols):
  
  # We pick the last sequence for each id in the test data
  seq_array_test_last = [data_df[data_df['EVENT_ID']==id][sequence_cols].values[-sequence_length:] 
                         for id in data_df['EVENT_ID'].unique() if len(data_df[data_df['EVENT_ID']==id]) >= sequence_length]

  seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
  # Similarly, we pick the labels
  
  y_mask = [len(data_df[data_df['EVENT_ID']==id]) >= sequence_length for id in data_df['EVENT_ID'].unique()]
  #print("y_mask")
  #print("y_mask:",y_mask)
  label_array_test_last = data_df.groupby('EVENT_ID')[targetColName].nth(-1)[y_mask].values
  label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
  
  return seq_array_test_last, label_array_test_last

# COMMAND ----------

def printKPIs(y_true_label_classes , y_pred_class):
  print("+++++Using True Classes[macro]+++++")
  precision,recall,fscore,support=score(y_true_label_classes , y_pred_class,average='macro')
  print('Precision : {}%'.format(round_down_float(precision*100,2)))
  print('Recall    : {}%'.format(round_down_float(recall*100,2)))
  print('F-score   : {}%'.format(round_down_float(fscore*100,2)))
  print('Support   : {}'.format(support))

  print("+++++Using True Classes[micro]+++++")
  precision,recall,fscore,support=score(y_true_label_classes , y_pred_class,average='micro')
  print('Precision : {}%'.format(round_down_float(precision*100,2)))
  print('Recall    : {}%'.format(round_down_float(recall*100,2)))
  print('F-score   : {}%'.format(round_down_float(fscore*100,2)))
  print('Support   : {}'.format(support))

  print("+++++Using True Classes[weighted]+++++")
  precision,recall,fscore,support=score(y_true_label_classes , y_pred_class,average='weighted')
  print('Precision : {}%'.format(round_down_float(precision*100,2)))
  print('Recall    : {}%'.format(round_down_float(recall*100,2)))
  print('F-score   : {}%'.format(round_down_float(fscore*100,2)))
  print('Support   : {}'.format(support))
  
  return precision,recall,fscore,support

# COMMAND ----------

def binaryEvaluation(model, x_seq_array,y_label_array):
  y_pred_class_prob = []
  y_pred_prob_threshold = []
  prob_threshold = 0.5
  y_pred_class = []
  y_true_array = []
  
  #print("y_label_array:", y_label_array)
  #y_label_array = np.argmax(y_label_array, axis=-1)
  #y_label_array = [ np.argmax(t) for t in y_label_array]
  y_pred_prop = model.predict(x_seq_array)
  
  
  for i in range(len(y_label_array)):
    if y_label_array[i]>0:
      y_true_array.insert(i,1)
    else:
      y_true_array.insert(i,0)
  
  for i in range(len(y_pred_prop)):
    
    if y_pred_prop[i]>prob_threshold:
      y_pred_class_prob.insert(i,1)
    else:
      y_pred_class_prob.insert(i,0)
    
    
  y_pred_class = y_pred_class_prob
  
  for i in range(len(y_pred_class)):
    y_pred_prob_threshold.insert(i,prob_threshold)
    ##print("Target=%s, Predicted=%s, Prob=%s" % (y_label_array[i], y_pred_class[i],y_pred_prop[i]))
  
  y_label_array = y_true_array
  print("y_label_array:", y_label_array)
  print("y_pred_class:",y_pred_class)
  
  cm = confusion_matrix(y_label_array, y_pred_class)
  
#   cm = confusion_matrix(y_target=y_label_array, 
#                           y_predicted=y_pred_class,
#                           binary=True)
  
  print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
  print(cm)
  
  tn, fp, fn, tp = confusion_matrix(y_label_array, y_pred_class).ravel()
  (tn, fp, fn, tp)
  
  report = classification_report(y_label_array, y_pred_class)
  print(report)
  
  precision,recall,fscore,support = printKPIs(y_label_array , y_pred_class)
  
  return cm, precision,recall,fscore, y_label_array, y_pred_class, y_pred_prop, y_pred_prob_threshold

def multiClassEvaluation(model, x_seq_array,y_label_array):
  y_pred_class_prob = []
  y_pred_prob_threshold = []
  prob_threshold = 0.5
  y_pred_class = []
  y_true_label = y_label_array
  
  try:
    #y_pred = model.predict_classes(valSeq_array)
    y_pred_prop = model.predict(x_seq_array)
    y_pred_class = np.argmax(x_seq_array, axis=1)
  except:
    print("Error CM1")
    pass
    
    try:
      y_true_label_classes = np.argmax(y_true_label, axis=-1)     
      try:
        
#         y_true_non_category = [ np.argmax(t) for t in y_true_label]
#         y_predict_non_category = [ np.argmax(t) for t in y_pred_class]
        
        print('[Exten ML] Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')

        conf_mat = confusion_matrix(y_target=y_true_label_classes, 
                          y_predicted=y_pred_class, 
                          binary=False)
        cm = conf_mat
        print(conf_mat)
        
        report = classification_report(np.argmax(y_true_label, axis=1), y_pred_class)
       
        print(report)
      except:
        print("Error CM1 and classification_report")
        pass
      try:
        precision,recall,fscore,support = printKPIs(y_true_label_classes , y_pred_class)
      except:
        print("Error in calculating F Score.")
        pass
    except:
      pass
    
    return cm, precision,recall,fscore, y_true_label_classes , y_pred_class, y_pred_prop, y_pred_prob_threshold

# COMMAND ----------

def evaluationMetrics(x_seq_array, y_true_label,isBinary,model):
  
  scores_test = model.evaluate(x_seq_array, y_true_label, verbose=0)
  print('Evaluation Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))
  print("Score: {} ".format(round_down_float(scores_test[0]*100,2)))
  
  #y_pred_class = model.predict_classes(valSeq_array)
  if (isBinary):
    return binaryEvaluation(model, x_seq_array,y_true_label)
  else:    
    return multiClassEvaluation(model, x_seq_array,y_true_label)

# COMMAND ----------

def isOptimal(history,countTrainSet,score,curVLoss,nEpoch):
  result=[]
  lossOptimal = False
  curLoss =0.01
  curMAcc = 0.90
  curVAcc = 0.90
  
  nMiniMumEpoch = 10
  optimalPoint = 3
  maxBatch = 35
  
  vLoss = history.history['val_loss']
  mLoss = history.history['loss']
  mAcc = history.history['acc']
  vAcc = history.history['val_acc']
    
  result.insert(0,mLoss[0])
  result.insert(1,mAcc[0])
  result.insert(2,vLoss[0])
  result.insert(3,vAcc[0])
  
  print("[Loss,Accuracy,Validation Loss, Validation Accuracy]")
  print(result)
    
  if countTrainSet >= maxBatch-1:
    if(vLoss[0] < curVLoss):
      score = score+1
    else:
      score = 0
  
  if score>=optimalPoint:
    lossOptimal = True
  else:
    curVLoss=vLoss[0]
    curLoss=mLoss[0]
    curMAcc=mAcc[0]
    curVAcc=vAcc[0]
    
  if (lossOptimal is True and countTrainSet >= maxBatch-1 and nEpoch>=nMiniMumEpoch):
    return True,score,result,curVLoss
  else:
    return False,score,result,curVLoss

# COMMAND ----------

def saveFileToDataLake(resultMetricDict):
  resultMetrixLSTM = pd.DataFrame.from_dict(resultMetricDict, orient='index', columns=['Epoch','m_code', 'Loss', 'Acc', 'Val_Loss', 'Val_Acc', 'Precision','Recall','F1' ])
  
  resultMetrixLSTM = resultMetrixLSTM.sort_values([ 'm_code'])

  print("%.2f%% (+/- %.2f%%)" % (np.mean(resultMetrixLSTM['Acc']), np.std(resultMetrixLSTM['Acc'])))

  resultMetrixLSTM.to_csv("result_BDLSTM_DL.csv")
  fromPathCSV = "file:/databricks/driver/result_BDLSTM_DL.csv"
  
  try:
    print("Copying file [",fromPathCSV,"] to Data Lake....")
    copyData(fromPathCSV, dataLake+"/",False)
  except:
    print("Error while trying to transfer file "+"file:/databricks/driver/"+fromPathCSV," to ",dataLake+"/")
    pass

# COMMAND ----------

def printProb(model,test_seq_array,test_label_array):
  # make a prediction
  y_proba = model.predict(test_seq_array)
  y_classes = model.predict_classes(test_seq_array)

  #ynew = keras.np_utils.probas_to_classes(y_proba)
  #p_prob=model.predict_proba(test_seq_array,batch_size=800,verbose=1)
  #y_classes = y_proba.argmax(axis=-1)
  ynew = y_classes
  ynewP = y_proba
  # show the inputs and predicted outputs

  for i in range(len(test_label_array)):
      print("X=%s, Predicted=%s, Prob=%s" % (test_label_array[i], ynew[i],ynewP[i]))

# COMMAND ----------

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

# COMMAND ----------

def train_hvd(modelCode,model, trainMap,valMap, mode,tf,learning_rate,min_max_scaler,isBinary,old_weights,startSet,startEpoch):
  tensor_board = TensorBoard(log_dir=tfb_log_dir, histogram_freq=1, write_graph=True, write_images=True)
  #isBinary = True
  
  if isBinary:
    classType = "BINARY"
    targetColName = "LABEL1"
  else:
    classType = "MULTI"
    targetColName = "LABEL2"
    
  if mode=="HRV":
    # Horovod: initialize Horovod.
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

  #   with tf.Graph().as_default():
  #     config = tf.ConfigProto(allow_soft_placement=True)
  #     config.gpu_options.visible_device_list = '0'

    K.set_session(tf.Session(config=config))

    # Horovod: adjust learning rate based on number of GPUs.
    v_optimizer = keras.optimizers.Adadelta(learning_rate * hvd.size())
    # Horovod: Wrap optimizer with Horovod DistributedOptimizer.
    v_optimizer = hvd.DistributedOptimizer(v_optimizer)
  
    # Horovod: Broadcast initial variable states from rank 0
    # to all other processes. This is necessary to ensure 
    # consistent initialization of all workers when training is
    # started with random weights or restored from a checkpoint.
    tensor_board = TensorBoard(log_dir=tfb_log_dir, histogram_freq=1, write_graph=True, write_images=True)
    callbacks = [tensor_board, hvd.callbacks.BroadcastGlobalVariablesCallback(0), keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')]
  
  
    #modelNameList = ["buildBinaryClassModel","buildMultipleClassModel","buildMultiAttentionModel"]
    #modelCodeMap = {"LSTM":"buildBinaryClassModel", "BDLSTM":"buildMultipleClassModel","BDLSTM_ATTEN":"buildMultiAttentionModel"}
  else:
    v_optimizer = keras.optimizers.Adam(lr=learning_rate)
    #v_optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    #v_optimizer = keras.optimizers.SGD(lr=learning_rate, clipvalue=1)
    #v_optimizer =  keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    callbacks = [tensor_board, 
                 #keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')
                 keras.callbacks.EarlyStopping(monitor='auc_roc', patience=10, verbose=1, mode='max')
                ]
  
  print("Start Train Model ",mode)
  
  cLoop = 1
  resultMetric = {}
  score = 0
  rolling_win_size = 30
  isNotSatisfacEval = True
  nEpochs = 10
  lossOptimal = False
  history =""
  score = 0
  cvscores = []
  curVLoss = 0.001
  resutl = []
  maxBatch = 10
  
  ## Multiple Classifications
  #val_label_array = to_categorical(val_label_array, num_classes=3, dtype='int32')
  
  if startEpoch>1:
    cLoop = startEpoch
  else:
    cLoop = 1
    
  for nEpoch in range(nEpochs):
    
    countTrainSet = 1
    trainDataSetKeys = trainMap.keys()
    
    #Hyperparameters
    v_batch_size = 200
    v_validation_split = 0.05
    v_verbose = 2
    
    #verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
    
    v_LSTMUnitLayer1 = 100
    v_LSTMUnitLayer2 = 50
    v_LSTMUnitLayer3 = 30
    
    v_Dropout = 0.2
    v_maxEpoch = 20
    scores_test = []
    
    for trainKey in trainDataSetKeys:
      if (trainKey>=startSet and nEpoch>=startEpoch):
        if isNotSatisfacEval is True:
          print("Starting Loop (cLoop) : ",str(cLoop))
          print("Train model using dataset {",str(trainKey),"}")
          isTrainSet = True
          train_df = getDataFromCSV(sqlContext, dbFSDir,trainMap[trainKey], selectedCols,isTrainSet,isBinary)
          
          ##Correct Sample Labels
          train_df = genSampleLabel(train_df)
          ##train_df = train_df.append(train_df)
          train_df = train_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
          train_df = add_features(train_df, rolling_win_size , sensor_cols)
          train_df = train_df.drop(columns=columns_to_drop)
          train_df,min_max_scaler = normalizeMaxMinTrain(train_df,min_max_scaler)
          train_df = train_df.sort_values(['EVENT_ID','CYCLE'])
          train_df = train_df.drop_duplicates(['EVENT_ID','CYCLE'], keep='last')
          seq_array, label_array, nb_features, nb_out = gen_data_train_val(targetColName, train_df,sequence_length, sequence_cols)
          
          valDataSetKeys = valMap.keys()
          
          if trainKey in valDataSetKeys:
            print("Loading Validate dataset[",trainKey,"]")
            val_df = getDataFromCSV(sqlContext, dbFSDir,valMap[trainKey], selectedCols,True,isBinary)
            val_df = val_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
            val_df = add_features(val_df, rolling_win_size , sensor_cols)
            val_df = genSampleLabel(val_df)
            val_df = val_df.drop(columns=columns_to_drop)
            val_df, min_max_scaler = normalizeMaxMinTrain(val_df,min_max_scaler)
            val_df = val_df.sort_values(['EVENT_ID','CYCLE'])
            val_df = val_df.drop_duplicates(['EVENT_ID','CYCLE'], keep='last')
            
            ## Verify all sample sequences
            val_seq_array, val_label_array, nb_features_val, nb_out_val = gen_data_train_val(targetColName, val_df,sequence_length, sequence_cols)
            ## Verify all sample sequences
            ##test_seq_array, test_label_array = gen_data_test_val(targetColName, val_df,sequence_length, sequence_cols)
            
          else:
            print("Error not found validation set matching with traing set number:",trainKey)
          
          printDFPortion(train_df, val_df, targetColName)
          
#           print("Finish Gen Train Data Sequence")
#           print("Finish Gen Validate Data Sequence")
          
          # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
          # Horovod: Save checkpoints only on worker 0 to prevent 
          # other workers from overwriting and corrupting them.
          ###checkpoint_dir = dataLake
          
          if mode=="HRV":
            if hvd.rank() == 0:
              callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_dir+ str(cLoop)+'checkpoint-hvd.hdf5', save_weights_only=True))
          
          #original_label_array = label_array
          
          #Multiple Classification
          #label_array = to_categorical(label_array, num_classes=3, dtype='int32')
#          val_label_array = to_categorical(val_label_array, num_classes=3, dtype='int32')
          nb_classes=label_array.shape[1]
          vb_classes=val_label_array.shape[1]          
#           print("label_array : nb_classes: ",nb_classes)
#           print("val_label_array : vb_classes: ",vb_classes)

          if len(old_weights)==0 and classType=="MULTI":
            model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features),merge_mode='concat'))
#             print("Created Bidirectional 1")
            model.add(Dropout(v_Dropout))
            model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2,return_sequences=True)))
#             print("Created Bidirectional 2")
            model.add(Dropout(v_Dropout))
            model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer3,return_sequences=False)))
#             print("Created Bidirectional 3")
            model.add(Dropout(v_Dropout))
            model.add(Dense(units=nb_classes,activation='softmax'))
          elif len(old_weights)==0 and classType=="BINARY":
            model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features),merge_mode='concat'))
            model.add(Dropout(v_Dropout))
#             print("Created Bidirectional 1")
            model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2, return_sequences=False))) 
            model.add(Dropout(v_Dropout))
#             print("Created Bidirectional 2")
            model.add(Dense(units=nb_out, activation='sigmoid'))
            print("nb_out:",nb_out)
          else:
            print("Model Already Constructed.")
          try:
            
            if old_weights!="":
              model.set_weights(old_weights)
              print("Reset weights successfully.")
              
          except:
            print("Failed reset weights.")
            pass
          
#           try:
#             model = multi_gpu_model(model,gpus=4)
#             print("Training using multiple GPUs..")
#           except:
#             print("Training using single GPU or CPU..")
#             pass
          
          if nb_classes>2:
            model.compile(loss='categorical_crossentropy', optimizer=v_optimizer, metrics=['accuracy'])
            print("set loss: categorical_crossentropy ")
          else:
            model.compile(loss='binary_crossentropy', optimizer=v_optimizer, metrics=['accuracy',auc_roc])
            print("set loss: binary_crossentropy ")
            
          print(model.summary())
          
          processCode = str(cLoop)+"_R_"+str(trainKey)
          
          if mode=="HRV":
            if hvd.rank() == 0:
                callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_dir + '/'+processCode+'_checkpoint-{epoch}.h5'))
          
          ### Utilizing Horovod
          history = model.fit(seq_array, label_array,
                              batch_size=v_batch_size,
                              epochs=v_maxEpoch, 
                              verbose=2,
                  validation_data=(val_seq_array, val_label_array)
                  ,#validation_split=v_validation_split,
                  callbacks = callbacks
                             )
          
              
          try:
            old_weights = model.get_weights()
            # evaluate the model
          except:
            print("Error get_weights !")
          
          # list all data in history
          print(history.history.keys())
          
          #val_seq_array, val_label_array = gen_data_test_val(targetColName, val_df,sequence_length, sequence_cols)
          #val_label_array = to_categorical(val_label_array, num_classes=3, dtype='int32')
          # cm,precision_test,recall_test,f1_test, y_true_label, y_predicted = evaluationMetrics(val_seq_array,val_label_array,isBinary,model)
          
          #printProb(model,val_seq_array , val_label_array)
          
          try:
            
            #cm,precision_test,recall_test,f1_test, y_true_label, y_predicted, y_pred_prop, y_pred_prob_thrldeshod = evaluationMetrics(val_seq_array,val_label_array,isBinary,model)
            cm,precision_test,recall_test,f1_test, y_true_label, y_pred_class, y_pred_prop, y_pred_prob_threshold = evaluationMetrics(val_seq_array, val_label_array,isBinary,model)
            
          except:
            precision_test = 0
            recall_test = 0
            f1_test=0
            print("Error in evaluation performance [evaluationMetrics]!")
            #return model
            pass
          
          if len(old_weights)==0:
              print("Error Empty Weights!!")
          else:
              print("Has weights!!")
          
          if mode!="HRV":
            try:
              currentModelPath = processCode + "_"+model_path
              print("Trying to save model : "+currentModelPath)
              model.save(currentModelPath)

              try:
                fromPath = "file:/databricks/driver/"+currentModelPath
                print("Copying file [",fromPath,"] to Data Lake....")
                copyData(fromPath, dataLake+"/model",False)
                
              except:
                print("Error while trying to transfer file "+"file:/databricks/driver/"+currentModelPath," to ",dataLake+"/model")
                pass  
                print("Model Saved >> ",currentModelPath)
            
            except:
              print("Error Saving Model",currentModelPath)
              pass
          
          try:
            lossOptimal, score, result, curVLoss = isOptimal(history,countTrainSet,score,curVLoss,nEpoch)
            #resultMetric[cLoop] = [cLoop, processCode] + result
            resultMetric[cLoop] = [cLoop, processCode] + result + [precision_test,recall_test,f1_test]
            print(resultMetric)
            saveFileToDataLake(resultMetric)
          except:
            print("Erro write metric file.")
            pass
            
          if lossOptimal is False:
            countTrainSet=countTrainSet+1
          else:
            break
          cLoop = cLoop+1
        else:
          print("Skip DataSet:",trainKey)
      else:
        print("Train and evaluation is satisfactory!")
        break
  return model

# COMMAND ----------

def genSampleLabel(data_df):
  # generate label columns for training data
  # we will only make use of "label1" for binary classification, 
  # while trying to answer the question: is a specific engine going to fail within w1 cycles?
  w1 = 1440
  w0 = 720
  
  print("w1:",w1)
  print("w0:",w0)
  
  data_df['LABEL1'] = np.where(data_df['RUL'] <= w1, 1, 0 )
  data_df['LABEL2'] = data_df['LABEL1']
  data_df.loc[data_df['RUL'] <= w0, 'LABEL2'] = 2
  
  return data_df

# COMMAND ----------

# print(normalized_val_df[normalized_val_df['EVENT_ID']>=3868])

# COMMAND ----------

# targetColName = "LABEL2"
# seq_array, label_array, nb_features, nb_out = gen_data_train_val(targetColName, normalized_val_df,100, sequence_cols)

# COMMAND ----------

# display(normalized_val_df[normalized_val_df['EVENT_ID']>=1000000].sort_values(by='RUL',ascending=True))

# COMMAND ----------

#printDFPortion(val_df, val_df, "LABEL1")

# COMMAND ----------

rolling_win_size = 60
min_max_scaler = preprocessing.MinMaxScaler()
isBinary = True

# COMMAND ----------

# test_seq_array, test_label_array = gen_data_test_val("LABEL1", test_df,sequence_length, sequence_cols)
# #test_seq_array, test_label_array = gen_data_test_val(targetColName, test_df,sequence_length, sequence_cols)

# COMMAND ----------

# test_df.columns

# COMMAND ----------

# print(test_seq_array[0:2])

# COMMAND ----------

def loadModel(modelPath):
  model = Sequential()
  
  if os.path.isfile(modelPath):
    model = load_model(modelPath)
    print("Model Loaded!")
  else:
    print("Model File Not Found!")
  return model

# COMMAND ----------

runMode = "RELOAD"
startSet = 8
startEpoch = 1

modelPath = "7_R_7_16_BDLSTM_RV1.h5"

old_weights =""

if runMode == "RELOAD":  
  a = "dbfs:/mnt/Exploratory/WCLD/16_BDLSTM_RV1/"+modelPath
  b = "file:/databricks/driver/model"
  try:
    dbutils.fs.mkdirs("file:/databricks/driver/model")
  except:
    print("Error in creating new folder.")
  copyData(a,b,False)
  model = loadModel(modelPath)
  try:
    old_weights = model.get_weights()
    # evaluate the model
  except:
    model = Sequential()
    print("Error get_weights !")
else:
  model = Sequential()
  startSet = 1
  startEpoch = 1

#hr = HorovodRunner(np=3)  
learning_rate=0.001

#hr.run(train_hvd("BDLSTM",model, trainMap,val_df,"HRV",tf,learning_rate))

##{BDLSTM-BI,BDLSTM-MULTI}

#model = hr.run(train_hvd("BDLSTM-BI",model, trainMap,valMap, "HRV",tf,learning_rate,min_max_scaler,isBinary,old_weights,startSet,startEpoch))
model = train_hvd("BDLSTM-BI",model, trainMap,valMap, "SINGLEGPU",tf,learning_rate,min_max_scaler,isBinary,old_weights,startSet,startEpoch)

# COMMAND ----------


##Test Sets:
totalEvents =0
testMap, totalEvents = loadDataSet("TEST")
# print("Total Events in Test Dataset : ",str(totalEvents))
# testMap = {30: [3868, 4163, 4178, 4242, 4381, 4411, 4452, 4466, 5137, 5163, 6647, 3899, 3987, 4200, 4421, 4433, 4472, 4475, 4478, 4484, 4575, 4628, 6398, 6777], 31: [3909, 3978, 4402, 4457, 4500, 4528, 4577, 4701, 6324, 6380, 6709, 3922, 3985, 4013, 4415, 4422, 4470, 4503, 4584, 4623, 4690, 4710, 4731, 4821], 32: [3942, 3986, 4295, 4517, 4560, 4799, 4820, 4824, 4827, 4918, 4968, 3924, 3938, 4065, 4146, 4532, 4553, 4600, 4741, 4800, 4869, 4873, 4889, 4935], 33: [4010, 4040, 4052, 4523, 4579, 4596, 4597, 4620, 4733, 4763, 4829, 4854, 4883, 4888, 4950, 4984, 5746, 5753, 5808, 5942, 6233, 4165, 4612, 4912], 34: [4642, 4676, 4774, 4867, 4932, 4947, 4953, 4989, 4992, 5001, 5012, 5027, 5799, 5806, 5999, 6001, 6012, 6027, 6041, 6058, 6070, 6127, 6510, 6770]}

# testMap = {1: [4010, 4040, 4052, 4523, 4579, 4596, 4597, 4620, 4733, 4763, 4829, 4854, 4883, 4888, 4950, 4984, 5746, 5753, 5808, 5942, 6233, 4165, 4612, 4912], 
#            2: [4642, 4676, 4774, 4867, 4932, 4947, 4953, 4989, 4992, 5001, 5012, 5027, 5799, 5806, 5999, 6001, 6012, 6027, 6041, 6058, 6070, 6127, 6510, 6770]}
# testMap = {1: [4010, 4040]}

print(testMap)
testDataSetKeys = testMap.keys()
rolling_win_size = 60
test_df = {}

for testKey in testDataSetKeys:
  print("Loading Test dataset[",testKey,"]",testMap[testKey])
  new_test_df = getDataFromCSV(sqlContext, dbFSDir,testMap[testKey], selectedCols,False,isBinary)
  new_test_df = new_test_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
  new_test_df = add_features(new_test_df, rolling_win_size , sensor_cols)

  if len(test_df)>0:
    test_df = test_df.append(new_test_df)
  else:
    test_df = new_test_df
  
  test_df = test_df.drop(columns=columns_to_drop)
  test_df, min_max_scaler = normalizeMaxMinTrain(test_df,min_max_scaler)
  test_df = test_df.sort_values(['EVENT_ID','CYCLE'])
  test_df = test_df.drop_duplicates(['EVENT_ID','CYCLE'], keep='last')
  break
  

# COMMAND ----------

def runChart(y_pred_test,y_true_test,y_pred_prob,threshold):
  fig_verify = plt.figure(figsize=(15, 5))
  #threshold = 0.5
  #seaborn-bright
  
  ax = plt.subplot(211)
  with plt.style.context('seaborn-bright'):
    plt.plot(y_pred_test, 'g-o')
    plt.plot(y_true_test, 'b-o')
  
  plt.title('prediction')
  plt.legend(['predicted', 'actual'], loc='upper right')
  #plt.xscale('symlog')
  plt.ylabel('Prediction(0/1)')
  plt.xlabel('event')
  plt.grid(False)
  #plt.gca().xaxis.grid(True, which='minor')  # minor grid on too
  #chartBox = ax.get_position()
  #ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*1.2, chartBox.height])
  #ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
  
  
  plt.subplot(212)
  with plt.style.context('seaborn-bright'):
    plt.plot(y_pred_prob, 'r-o')
    plt.plot(threshold,'limegreen')
  
  plt.title('probability')
  plt.legend(['value','threshold'], loc='upper right')
  plt.ylabel('raw result')
  plt.xlabel('event')
  plt.grid(True)
  
  #plt.gca().xaxis.grid(True, which='minor')  # minor grid on too
  #chartBox2 = ax2.get_position()
  #ax2.set_position([chartBox2.x0, chartBox2.y0, chartBox2.width*0.6, chartBox2.height])
  #ax2.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
  
  plt.tight_layout()
  
  datafile = "model_verify.png"
  fig_verify.savefig(datafile)

# COMMAND ----------

# display(test_df[test_df['EVENT_ID']==3868])

# COMMAND ----------

targetColName = "LABEL1"
test_seq_array, test_label_array = gen_data_test_val(targetColName, test_df,sequence_length, sequence_cols)
nb_classes=test_label_array.shape[1]

#test_seq_array, test_label_array, nb_features, nb_out = gen_data_train_val(targetColName, test_df,sequence_length, sequence_cols)

# try:
  
#   #test_label_array = keras.utils.to_categorical(test_label_array, num_classes=1, dtype='int32')
  
#   print("label_array : nb_classes: ",nb_classes)
#   isBinary = True
#   cm,precision_test,recall_test,f1_test, y_true_label, y_predicted = evaluationMetrics(test_seq_array,test_label_array,isBinary,model)
  
# except:
#   print("Error in evaluation performance!")

print("label_array : nb_classes: ",nb_classes)
isBinary = True

cm,precision_test,recall_test,f1_test, y_true_label, y_pred_class ,y_pred_prop, y_pred_prob_threshold = evaluationMetrics(test_seq_array,test_label_array,isBinary,model)

# COMMAND ----------

#cm, precision,recall,fscore, y_true_label, y_pred_class, y_pred_prop, y_pred_prob_threshold = binaryEvaluation(model, test_seq_array,test_label_array)

# COMMAND ----------

try:
  fig, ax = plot_confusion_matrix(conf_mat=cm,
                                  colorbar=True,
                                  show_absolute=True,
                                  show_normed=True)
except:
  cm = confusion_matrix(y_target=y_true_label, 
                y_predicted=y_pred_class,
                binary=isBinary)
  fig, ax = plot_confusion_matrix(conf_mat=cm)
  pass
display()

# COMMAND ----------

runChart(y_pred_class,y_true_label,y_pred_prop,y_pred_prob_threshold)
display()

# COMMAND ----------

# from sklearn.metrics import confusion_matrix
# cmx = confusion_matrix(y_true_label, y_predicted, labels=[0, 1])
# print(cmx)
# tn, fp, fn, tp = confusion_matrix(y_true_label, y_predicted, labels=[0, 1]).ravel()
# (tn, fp, fn, tp)

# COMMAND ----------

# report = classification_report(y_true_label, y_predicted)
# print(report)

# COMMAND ----------

# try:
#   print(resultMetric)
# except:
#   pass

# COMMAND ----------

### End Experiment ######