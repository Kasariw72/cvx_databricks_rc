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

from keras.metrics import categorical_accuracy as metrics
from mlxtend.evaluate import confusion_matrix 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report
import sys
import traceback
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_fscore_support as score
from keras.utils.np_utils import to_categorical

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
sparkAppName = "12_BDLSTM_RV6"
# define path to save model
model_path = sparkAppName+".h5"

dirPath = "file:/databricks/driver/"
dataLake = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
mntDatalake = "/mnt/Exploratory/WCLD/"+sparkAppName

tf_at_dl_dir = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
dbFSDir = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV3"
tensorboardLogPath = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName+"/tensorboard_log"
indexFilePath = "dbfs:/mnt/Exploratory/WCLD/dataset/EVENT_ID_DATA_SETS_FULL_V8.csv"
dfDirDBFS = dbutils.fs.ls(dbFSDir)
timestamp = datetime.datetime.now().strftime("%Y%m%d%H")

expContext = sparkAppName+"_"+str(timestamp)
tfb_log_dir = tfb_log_dir+expContext
print(expContext)
# dbutils.fs.mkdirs(tfb_log_dir)
### Enable Tensorboard and save to the localtion
#dbutils.tensorboard.start(tfb_log_dir)

# COMMAND ----------

def copyData(fromPath, toPath,recurseFlag=True):
  dbutils.fs.cp(fromPath, toPath,recurse=recurseFlag)
  print("Completed copying ["+fromPath+"] to "+toPath)

# COMMAND ----------

# print(expContext)
# dbutils.fs.mkdirs(dirPath+"/"+expContext)
# print(dirPath+expContext)
# dbutils.fs.mkdirs("dbfs:/RC/experiments/"+expContext)
# print("dbfs:/RC/experiments/"+expContext)
# expDir = "dbfs:/RC/experiments/"+expContext

# COMMAND ----------

selectedCols = ['CODE','YEAR','EVENT_ID','CYCLE', 'RUL','LABEL1','LABEL2','NOM_CYCLE','RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_LUBE_OIL_PRESS',
 'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',  'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP', 'LUBE_OIL_PRESS',  'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP', 'THROW_1_SUC_PRESS', 'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS', 'THROW_2_DISC_TEMP', 'THROW_2_SUC_PRESS', 'THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS', 'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP',
'CYL_6_TEMP', 'CYL_7_TEMP', 'CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP', 'CYL_12_TEMP', 'LUBE_OIL_PRESS_ENGINE', 'MANI_PRESS', 'RIGHT_BANK_EXH_TEMP', 'SPEED',
'VIBRA_ENGINE']

sensor_cols = ['RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_LUBE_OIL_PRESS',
 'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',  'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP', 'LUBE_OIL_PRESS',  'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP', 'THROW_1_SUC_PRESS', 'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS', 'THROW_2_DISC_TEMP', 'THROW_2_SUC_PRESS', 'THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS', 'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP',
'CYL_6_TEMP', 'CYL_7_TEMP', 'CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP', 'CYL_12_TEMP', 'LUBE_OIL_PRESS_ENGINE', 'MANI_PRESS', 'RIGHT_BANK_EXH_TEMP', 'SPEED',
'VIBRA_ENGINE']

selectedCols = ['CODE','YEAR','EVENT_ID','CYCLE', 'RUL','LABEL1','LABEL2', 'RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_LUBE_OIL_PRESS',
 'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',  'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP', 'LUBE_OIL_PRESS',  'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP', 'THROW_1_SUC_PRESS', 'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS', 'THROW_2_DISC_TEMP', 'THROW_2_SUC_PRESS', 'THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS', 'THROW_3_DISC_TEMP',
'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP',
'CYL_6_TEMP', 'CYL_7_TEMP', 'CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP', 'CYL_12_TEMP', 'LUBE_OIL_PRESS_ENGINE', 'MANI_PRESS', 'RIGHT_BANK_EXH_TEMP', 'SPEED',
'VIBRA_ENGINE', 'S1_RECY_VALVE', 'S1_SUCT_PRESS', 'S1_SUCT_TEMPE', 'S2_STAGE_DISC_PRESS', 'S2_SCRU_SCRUB_LEVEL', 'GAS_LIFT_HEAD_PRESS', 'IN_CONT_CONT_VALVE', 'IN_SEP_PRESS', 'WH_PRESSURE']

sensor_cols = ['LUBE_OIL_PRESS',  'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP', 'THROW_1_SUC_PRESS', 
               'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS', 'THROW_2_DISC_TEMP', 'THROW_2_SUC_PRESS', 'THROW_2_SUC_TEMP', 
               'THROW_3_DISC_PRESS', 'THROW_3_DISC_TEMP','THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 
               'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 
               'CYL_4_TEMP', 'CYL_5_TEMP','CYL_6_TEMP', 'CYL_7_TEMP', 'CYL_8_TEMP',
               'CYL_9_TEMP', 'CYL_10_TEMP','CYL_11_TEMP', 'CYL_12_TEMP', 'LUBE_OIL_PRESS_ENGINE', 
               'MANI_PRESS', 'RIGHT_BANK_EXH_TEMP','SPEED','VIBRA_ENGINE', 'S1_RECY_VALVE', 
               'S1_SUCT_PRESS', 'S1_SUCT_TEMPE', 'S2_STAGE_DISC_PRESS', 'S2_SCRU_SCRUB_LEVEL', 'GAS_LIFT_HEAD_PRESS', 
               'IN_CONT_CONT_VALVE', 'IN_SEP_PRESS', 'WH_PRESSURE']

resCols = ['RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_LUBE_OIL_PRESS',
 'RESI_LL_SPEED', 'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',  'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP']
# propertyCols = ['CODE','YEAR','EVENT_ID','CYCLE', 'RUL','LABEL1','LABEL2','NOM_CYCLE','BF_SD_TYPE']
# propertyCol2 = ['CODE','YEAR','(EVENT_ID + 10000000) as EVENT_ID','CYCLE', 'RUL','LABEL1','LABEL2','NOM_CYCLE']
# dupCol = propertyCol2 + sensor_cols


propertyCols = ['CODE','YEAR','CYCLE', 'RUL','LABEL1','LABEL2']

intCols = ['YEAR','CYCLE', 'RUL','LABEL1','LABEL2','EVENT_ID']

propertyCol0 = ['(EVENT_ID + 10000000) as EVENT_ID']
propertyCol2 = ['(EVENT_ID + 20000000) as EVENT_ID']
propertyCol1 = ['(EVENT_ID + 30000000) as EVENT_ID']

#sequence_cols = sensor_cols + sensor_av_cols + sensor_sd_cols
dupColLabel0 = propertyCols + resCols + propertyCol0 + sensor_cols
dupColLabel2 = propertyCols + resCols + propertyCol2 + sensor_cols
dupColLabel1 = propertyCols + resCols + propertyCol1 + sensor_cols
columns_to_drop = ['CODE','YEAR']

sequence_cols = sensor_cols

# COMMAND ----------

print(sensor_cols)

# COMMAND ----------

#3 - Setup SparkSession(SparkSQL)
spark = (SparkSession.builder.appName(sparkAppName).getOrCreate())
print(spark)
#4 - Read file to spark DataFrame
from pyspark.sql import SQLContext
sqlContext = SQLContext(spark)

# COMMAND ----------

# MAGIC %md #Mini Batch Spark Data Loading

# COMMAND ----------

def upsampling(selectColDF,isTrain):
  if isTrain:
    newUpDF2 = selectColDF.where("RUL BETWEEN 1 AND 720")
    newUpDF2 = newUpDF2.selectExpr(dupColLabel2)
    #Upsampling Positive Class Label 1
    newUpDF1 = selectColDF.where("RUL BETWEEN 721 AND 1440")
    newUpDF1 = newUpDF1.selectExpr(dupColLabel1)
    return newUpDF2.union(newUpDF1)
  else:
    #Upsampling Positive Class Label 1
    newUpDF1 = selectColDF.where("RUL BETWEEN 721 AND 1440")
    newUpDF1 = newUpDF1.selectExpr(dupColLabel1)
    #Upsampling Negative Class Label 0
    newUpDF0 = selectColDF.where("RUL BETWEEN 1441 AND 2602")
    newUpDF0 = newUpDF0.selectExpr(dupColLabel0)
    return newUpDF1.union(newUpDF0)

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

def getDataFromCSV(sqlContext, dbFSDir,eventIds, selectedCols, isTrainSet):
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
  selectColDF = spark.sql("SELECT * FROM RC_DATA_TMP WHERE RUL BETWEEN 1 AND 3200 AND RUL>0 ORDER BY YEAR, EVENT_ID, CYCLE")
  selectColDF = selectColDF.dropDuplicates()
  ## Up positive samples X2
  #return selectColDF
  
  upsamplingDF = upsampling(selectColDF,isTrainSet)
  resultDF = selectColDF.union(upsamplingDF)
  resultDF = resultDF.dropDuplicates(['EVENT_ID','CYCLE','RUL'])
  #resultPD = resultDF.toPandas()
  
  #resultDF = selectColDF
  #selectDataCols = selectDataCols.sort_values(['YEAR','MONTH','DAY','EVENT_ID','CYCLE'])
  return resultDF.toPandas()

# COMMAND ----------

def gen_data_test_val(targetColName, test_df,sequence_length, sequence_cols):
  
  # We pick the last sequence for each id in the test data
  seq_array_test_last = [test_df[test_df['EVENT_ID']==id][sequence_cols].values[-sequence_length:] 
                         for id in test_df['EVENT_ID'].unique() if len(test_df[test_df['EVENT_ID']==id]) >= sequence_length]

  seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
  # Similarly, we pick the labels
  
  y_mask = [len(test_df[test_df['EVENT_ID']==id]) >= sequence_length for id in test_df['EVENT_ID'].unique()]
  #print("y_mask")
  #print("y_mask:",y_mask)
  label_array_test_last = test_df.groupby('EVENT_ID')[targetColName].nth(-1)[y_mask].values
  label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
  
  return seq_array_test_last, label_array_test_last

# COMMAND ----------

## Loading Index Data Table
df2 = (sqlContext.read.option("header","true").option("inferSchema", "true").csv(indexFilePath))
# If the path don't have file:/// -> it will call hdfs instead of local file system
df2.cache()
print("Spark finishs caching:"+indexFilePath)

# COMMAND ----------

#compute dataframe using sql command via string
df2.createOrReplaceTempView("INDEX_RC_DATA")
sparkDF = spark.sql("SELECT * FROM INDEX_RC_DATA")

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

def copyData(fromPath, toPath,recurseFlag=True):
  dbutils.fs.cp(fromPath, toPath,recurse=recurseFlag)
  print("Completed copying ["+fromPath+"] to "+toPath)
  
def round_down_float(n, decimals=0):
    multiplier = 10 ** decimals
    return float(math.floor(n * multiplier) / multiplier)

# COMMAND ----------

def normalizeMaxMinTrain(sampleDataSet,min_max_scaler):
# MinMax normalization (from 0 to 1)
  #sampleDataSet['NOM_CYCLE'] = sampleDataSet['CYCLE']
  cols_normalize = sampleDataSet.columns.difference(['EVENT_ID','YEAR','CYCLE','RUL','LABEL1','LABEL2'])
  norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(sampleDataSet[cols_normalize]), 
                               columns=cols_normalize, 
                               index=sampleDataSet.index)
  join_df = sampleDataSet[sampleDataSet.columns.difference(cols_normalize)].join(norm_train_df)
  sampleDataSet = join_df.reindex(columns = sampleDataSet.columns)
  print("Finish normalization train dataset!")
  return sampleDataSet

# COMMAND ----------

# MAGIC %md ## Deep Learining - LSTM for Binary Classification Objective

# COMMAND ----------

# pick a large window size of 120 cycles
# Target 1 days using 120 time steps

sequence_length = 100
input_length = 100

sensor_av_cols = ["AVG_" + nm for nm in sensor_cols]
sensor_sd_cols = ["SD_" + nm for nm in sensor_cols]

sequence_cols1 = sensor_av_cols + sensor_sd_cols
sequence_cols = sequence_cols + sequence_cols1

#print(sequence_cols)
columns_to_drop = ['CODE','YEAR']
dbFSDir = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV6"

v_batch_size = 200
v_validation_split = 0.05
v_verbose = 2
#verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
  
v_LSTMUnitLayer1 = 120
v_LSTMUnitLayer2 = 60
v_Dropout = 0.2
v_maxEpoch = 1

# COMMAND ----------

print(sequence_cols)

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

def genSampleLabel(data_df):
  # generate label columns for training data
  # we will only make use of "label1" for binary classification, 
  # while trying to answer the question: is a specific engine going to fail within w1 cycles?
  w1 = 1440
  w0 = 720
  
  data_df['LABEL1'] = np.where(data_df['RUL'] <= w1, 1, 0 )
  data_df['LABEL2'] = data_df['LABEL1']
  data_df.loc[data_df['RUL'] <= w0, 'LABEL2'] = 2
  
  return data_df

# COMMAND ----------

def buildBinaryClassModel(algoName,old_weights,v_LSTMUnitLayer1,v_LSTMUnitLayer2,sequence_length,nb_features,nb_out,v_Dropout,v_maxEpoch,v_batch_size,seq_array, label_array,valSeq_array, valLabel_array,model):
  
  print("nb_out:",nb_out)
  
  if len(old_weights)==0:
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features),merge_mode='concat'))
    print("Created Bidirectional 1")
    model.add(Dropout(v_Dropout))
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2, return_sequences=False)))
    print("Created Bidirectional 2")
    model.add(Dropout(v_Dropout))
    model.add(Dense(units=nb_out, activation='sigmoid'))
    
  else:
    try:
      model.set_weights(old_weights)
      print("Reset weights successfully.")
    except:
      print("Failed resetting weights.")
      pass
  try:
    model = multi_gpu_model(model,gpus=4, cpu_merge=False)
    print("Training using multiple GPUs..")
  except:
    print("Training using single GPU or CPU..")
    pass
  
  v_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  
  model.compile(loss='binary_crossentropy', optimizer=v_optimizer, metrics=['accuracy'])
  
  print(model.summary())
  
  history = model.fit(seq_array, label_array,
          batch_size=v_batch_size, epochs=v_maxEpoch, shuffle=False, verbose=2,
          #validation_split=0.05,
          validation_data=(valSeq_array, valLabel_array),
          callbacks = [tensor_board, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                         keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
  
  try:
    
    scores_test = model.evaluate(valSeq_array, valLabel_array, verbose=0)
    print("scores_test:",scores_test)
    y_true = valLabel_array
    y_pred = model.predict(valSeq_array)
    labels = np.argmax(y_pred, axis=-1)
    
    print("labels:",labels[0:20])
    lb = preprocessing.LabelBinarizer()
    lb.fit_transform(labels)
    print("lb:",lb)
    print("y_true[0:20]", y_true[0:20])
    print("y_pred[0:20]", y_pred[0:20])
    
    print('Evaluation Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))
    
    try:
      
      y_true_non_category = [ np.argmax(t) for t in y_true]
      y_predict_non_category = [ np.argmax(t) for t in y_pred ]
      
#       print("y_true_non_category[0:20]", y_true_non_category[0:20])
#       print("y_predict_non_category[0:20]", y_predict_non_category[0:20])
      
      conf_mat = confusion_matrix(y_target=y_true_non_category, 
                      y_predicted=y_predict_non_category, 
                      binary=True)
  
      print(conf_mat)
    
    except:
      print("Error CM2")
      pass
    
    try:
      fbeta = fbeta_score(y_true_non_category, y_predict_non_category, beta=1)
      print('fbeta :',fbeta)
    except:
      print("Error in calculating F Score.")
      pass
    
    try:
      
      predicted = np.argmax(y_pred, axis=1)
      report = classification_report(np.argmax(y_true, axis=1), predicted)
      print(report)

    except:
      print("y_pred:",y_pred[0:20])
      print("Error in calculating top_k_acc.")
      pass
    
  except:
    pass
  
  return model, history

# COMMAND ----------

def buildMultipleClassModel(algoName,old_weights,v_LSTMUnitLayer1,v_LSTMUnitLayer2,sequence_length,nb_features,nb_out,v_Dropout,v_maxEpoch,v_batch_size,seq_array, label_array,valSeq_array, valLabel_array,model):
  
  label_array = keras.utils.to_categorical(label_array, num_classes=3, dtype='int32')
  valLabel_array = keras.utils.to_categorical(valLabel_array, num_classes=3, dtype='int32')
  
#   print("after label_array:",valLabel_array[0:5])
#   print("valLabel_array", valLabel_array[0:5])
  nb_classes=valLabel_array.shape[1]
  print("nb_classes:",nb_classes)
  
  if len(old_weights)==0:
    #model.add(Embedding(sequence_length, nb_features, input_length=input_length))
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features)))
    print("Created Bidirectional 1")
    model.add(Dropout(v_Dropout))
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2, return_sequences=True)))
    print("Created Bidirectional 2")
    model.add(Dropout(v_Dropout))
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2)))
    print("Created Bidirectional 3")
    model.add(Dropout(v_Dropout))
    #model.add(Flatten())
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
  else:
    try:
      model.set_weights(old_weights)
      print("Reset weights successfully.")
    except:
      print("Failed resetting weights.")
      pass
  try:
    model = multi_gpu_model(model,gpus=4, cpu_merge=False)
    print("Training using multiple GPUs..")
  except:
    print("Training using single GPU or CPU..")
    pass
#   model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features)))
#   model.add(Dropout(v_Dropout))
#   model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2)))
#   model.add(Dropout(v_Dropout))
#   model.add(Dense(units=nb_classes))
#   model.add(Activation('softmax'))
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=my_metrics)
  
  #v_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  v_optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
  model.compile(loss='categorical_crossentropy', optimizer=v_optimizer, metrics=['accuracy'])
  
  #'categorical_accuracy'
  #model.compile(loss='sparse_categorical_crossentropy', optimizer=v_optimizer, metrics=['accuracy'])
  print(model.summary())
  
  history = model.fit(seq_array, label_array,
          batch_size=v_batch_size, epochs=v_maxEpoch, shuffle=False, verbose=2,
          #validation_split=0.05,
          validation_data=(valSeq_array, valLabel_array),
          callbacks = [tensor_board, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                         keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
  
  try:
    
    scores_test = model.evaluate(valSeq_array, valLabel_array, verbose=0)
    print("scores_test:",scores_test)
    y_true = valLabel_array
    y_pred = model.predict(valSeq_array)
    labels = np.argmax(y_pred, axis=-1)
    
    print("labels:",labels[0:20])
    lb = preprocessing.LabelBinarizer()
    lb.fit_transform(labels)
    print("lb:",lb)
    print("y_true[0:20]", y_true[0:20])
    print("y_pred[0:20]", y_pred[0:20])
    
    print('Evaluation Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))
    
    try:
      
      y_true_non_category = [ np.argmax(t) for t in y_true]
      y_predict_non_category = [ np.argmax(t) for t in y_pred ]
      
#       print("y_true_non_category[0:20]", y_true_non_category[0:20])
#       print("y_predict_non_category[0:20]", y_predict_non_category[0:20])
      
      conf_mat = confusion_matrix(y_target=y_true_non_category, 
                      y_predicted=y_predict_non_category, 
                      binary=False)
  
      print(conf_mat)
    
    except:
      print("Error CM2")
      pass
    
    try:
      fbeta = fbeta_score(y_true_non_category, y_predict_non_category, beta=1)
      print('fbeta :',fbeta)
    except:
      print("Error in calculating F Score.")
      pass
    
    try:
      
      predicted = np.argmax(y_pred, axis=1)
      report = classification_report(np.argmax(y_true, axis=1), predicted)
      print(report)

    except:
      print("y_pred:",y_pred[0:20])
      print("Error in calculating top_k_acc.")
      pass
    
  except:
    pass
  
  return model, history

# COMMAND ----------

def trainModel(modelCode, old_weights,train_df,val_df,test_df,iterNum,tensor_board,model):
  
  modelNameList = ["buildBinaryClassModel","buildMultipleClassModel","buildMultiAttentionModel"]
  modelCodeMap = {"LSTM":"buildBinaryClassModel", "BDLSTM":"buildMultipleClassModel","BDLSTM_ATTEN":"buildMultiAttentionModel"}
  
  print("Start Train Model ",modelCode)
  
  targetColName = "LABEL1"
  
  if modelCodeMap[modelCode] == "buildMultipleClassModel":
    targetColName = "LABEL2"
  elif modelCodeMap[modelCode]  == "buildMultiAttentionModel":
    targetColName = "LABEL2"
  else:
    modelCodeMap[modelCode]  = "LABEL1"
  
  ##print(sequence_cols)
  seq_array, label_array, nb_features, nb_out = gen_data_train_val(targetColName, train_df,sequence_length, sequence_cols)
  print("Finish Gen Train Data Sequence")
  
  valSeq_array, valLabel_array, valNb_features, valNb_out = gen_data_train_val(targetColName, val_df,sequence_length, sequence_cols)
  print("Finish Gen Validate Data Sequence")
  
  model, history = buildBinaryClassModel (modelCode,
                                                       old_weights,
                                                       v_LSTMUnitLayer1,
                                                       v_LSTMUnitLayer2,
                                                       sequence_length,
                                                       nb_features,nb_out,
                                                       v_Dropout,
                                                       v_maxEpoch,
                                                       v_batch_size,
                                                       seq_array, 
                                                       label_array,
                                                       valSeq_array, 
                                                       valLabel_array,model)
  
  try:
    ##print(history.history.keys())
    old_weights = model.get_weights()
    # evaluate the model
    currentModelPath = str(iterNum)+"_"+model_path
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
  
  return history,old_weights, model

# COMMAND ----------

#Training Sets
totalEvents = 10
#trainMap,totalEvents = loadDataSet("TRAIN")
print("Total Events in Traing Dataset : ",str(totalEvents))
# # trainMap = {1: [2091, 2101], 
# #  2: [712, 714], 
# #  3: [924, 950]}
trainMap = {1: [2091, 2101, 2103, 2104, 2119, 538, 546, 574, 578, 580, 660, 672, 679, 693, 694, 704, 705, 707, 712, 714, 732, 780, 787, 791, 815, 819, 842, 849, 852, 857, 860, 863]}
print(trainMap)

#Validation Sets
#valMap , totalEvents = loadDataSet("VALI")
valMap ={23: [2734, 2753, 3060, 3093, 3129, 3134, 3533, 3559, 3755, 3772, 3774, 3781, 3793, 3812, 3826, 3837, 4373, 4445, 4449, 4641, 4772, 4851, 4955, 4969], 24: [2771, 2789, 2853, 3140, 3164, 3206, 3271, 3453, 3631, 3658, 3665, 3801, 3824, 3830, 3839, 3872, 3859, 3886, 4033, 4468, 4489, 5007, 6009, 6474]}

print("Total Events in Validation Dataset : ",str(totalEvents))
# valMap = {1: [635, 655, 760, 776]}
# valMap = {2: [660, 672, 679, 693, 694, 704, 705, 707, 712, 714, 732, 735, 740, 750, 751, 779, 834, 737, 743, 746], 3: [780, 787, 791, 815, 819, 842, 849, 852, 857, 860, 863, 878, 907, 912, 917, 926, 924, 950, 953, 961], 4: [1012, 1016, 1027, 1030, 1032, 1038, 1036, 1041, 1047, 1050, 1055, 1059, 1062, 1065, 1069, 1074, 1086, 1088, 1097, 1098], 5: [1115, 1120, 1122, 1128, 1133, 1134, 1144, 1149, 1162, 1165, 1172, 1174, 1189, 1228, 1229, 1251, 1259, 1262, 1187, 1202], 6: [1215, 1222, 1226, 1260, 1261, 1272, 1277, 1280, 1292, 1293, 1299, 1309, 1310, 1316, 1317, 1321, 1322, 1325, 1327, 1337]}

valMap = {2: [660, 672, 679, 693, 694, 704, 705, 707, 712, 714, 732, 735, 740, 750, 751, 779, 834, 737, 743, 746]}

print(valMap)

# #Test Sets:
#testMap,totalEvents = loadDataSet("TEST")
print("Total Events in Test Dataset : ",str(totalEvents))
# testMap = {30: [3868, 4163, 4178, 4242, 4381, 4411, 4452, 4466, 5137, 5163, 6647, 3899, 3987, 4200, 4421, 4433, 4472, 4475, 4478, 4484, 4575, 4628, 6398, 6777], 31: [3909, 3978, 4402, 4457, 4500, 4528, 4577, 4701, 6324, 6380, 6709, 3922, 3985, 4013, 4415, 4422, 4470, 4503, 4584, 4623, 4690, 4710, 4731, 4821], 32: [3942, 3986, 4295, 4517, 4560, 4799, 4820, 4824, 4827, 4918, 4968, 3924, 3938, 4065, 4146, 4532, 4553, 4600, 4741, 4800, 4869, 4873, 4889, 4935], 33: [4010, 4040, 4052, 4523, 4579, 4596, 4597, 4620, 4733, 4763, 4829, 4854, 4883, 4888, 4950, 4984, 5746, 5753, 5808, 5942, 6233, 4165, 4612, 4912], 34: [4642, 4676, 4774, 4867, 4932, 4947, 4953, 4989, 4992, 5001, 5012, 5027, 5799, 5806, 5999, 6001, 6012, 6027, 6041, 6058, 6070, 6127, 6510, 6770]}

testMap = {30: [3868, 4163, 4178, 4242, 4381, 4411, 4452, 4466, 5137, 5163, 6647, 3899, 3987, 4200, 4421, 4433, 4472, 4475, 4478, 4484, 4575, 4628, 6398, 6777]}
print(testMap)

# COMMAND ----------

def printDFPortion(train_df, val_df, label):
  
  if label=="LABEL2":
    try:
      print("+++++Train Data Set LABEL2 ++++++++++++")
      print(train_df['LABEL2'].value_counts())
      print('\nClass 0 samples = {0:.0%}'.format(train_df['LABEL2'].value_counts()[0]/train_df['LABEL2'].count()))
      print('\nClass 1 samples = {0:.0%}'.format(train_df['LABEL2'].value_counts()[1]/train_df['LABEL2'].count()))
      print('\nClass 2 samples = {0:.0%}'.format(train_df['LABEL2'].value_counts()[2]/train_df['LABEL2'].count()))
    except:
      pass
    print("+++++Validation Data Set LABEL2 ++++++++++++")
    try:
      print(val_df['LABEL2'].value_counts())
      print('\nClass 0 samples = {0:.0%}'.format(val_df['LABEL2'].value_counts()[0]/val_df['LABEL2'].count()))
      print('\nClass 1 samples = {0:.0%}'.format(val_df['LABEL2'].value_counts()[1]/val_df['LABEL2'].count()))
      print('\nClass 2 samples = {0:.0%}'.format(val_df['LABEL2'].value_counts()[2]/val_df['LABEL2'].count()))
    except:
      pass
  else:
    try:
      print("+++++Train Data Set LABEL1 ++++++++++++")
      # print stat for binary classification label
      print(train_df['LABEL1'].value_counts())
      print('\nNegaitve samples [0] = {0:.0%}'.format(train_df['LABEL1'].value_counts()[0]/train_df['LABEL1'].count()))
      print('\nPosiitve samples [1] = {0:.0%}'.format(train_df['LABEL1'].value_counts()[1]/train_df['LABEL1'].count()))
    except:
      pass

    try:
      print("+++++Validation Data Set LABEL1 ++++++++++++")
      print(val_df['LABEL1'].value_counts())
      print('\nNegaitve samples [0] = {0:.0%}'.format(val_df['LABEL1'].value_counts()[0]/val_df['LABEL1'].count()))
      print('\nPosiitve samples [1] = {0:.0%}'.format(val_df['LABEL1'].value_counts()[1]/val_df['LABEL1'].count()))
    except:
      pass
  

# COMMAND ----------

## Aj. Suggested to use only one set of UCM event to test model
maxBatch = len(trainMap)
val_df = ""
isNotSatisfacEval = True
old_weights =""

min_max_scaler = preprocessing.MinMaxScaler()
#quantile_scaler = preprocessing.QuantileTransformer(output_distribution='uniform')
tensor_board = TensorBoard(log_dir=tfb_log_dir, histogram_freq=1, write_graph=True, write_images=True)

model = Sequential()
test_df=""
val_df=""
nEpochs = 16
lossOptimal = False
history =""
score = 0
optimalPoint = 3
cvscores = []
curVLoss = 0.01

cLoop = 0
trainedSets = {}
validatedSets = {}

test_df={}
val_df={}
resultMetric = {}
score = 0
rolling_win_size = 30
nEpoch = 1
cLoop =1

# COMMAND ----------

def isOptimal(history,countTrainSet,score,curVLoss,nEpoch):
  result=[]
  lossOptimal = False
  curLoss =0.001
  curMAcc = 0.90
  curVAcc = 0.90
  nMiniMumEpoch = 10
  
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

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/12_BDLSTM_RV4/model/289_R_25_12_BDLSTM_RV4.h5 file:/databricks/driver

# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------

model = Sequential()
lastModelPath = "289_R_25_12_BDLSTM_RV4.h5"
old_weights = ""

if os.path.isfile(lastModelPath):
  ##estimator = load_model(modelPath)
  model = load_model(lastModelPath)
  #old_weights = model.get_weights()
  print("Model Loaded!",lastModelPath)
else:
  print("Model File Not Found!",lastModelPath)

# COMMAND ----------

# for nEpoch in range(nEpochs):
  
#   print("Starting Loop : ",str(cLoop))
#   countTrainSet = 1
#   trainDataSetKeys = trainMap.keys()
#   valDataSetKeys = valMap.keys()
  
#   for trainKey in trainDataSetKeys:
#     if (trainKey>=0 and nEpoch>=0):
#       cLoop = cLoop+1
#       if isNotSatisfacEval is True:
#         print("Train model using dataset {",str(trainKey),"}")
#         train_df = getDataFromCSV(sqlContext, dbFSDir,trainMap[trainKey], selectedCols)
#         train_df = train_df.append(train_df)
#         train_df = train_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
#         train_df = add_features(train_df, rolling_win_size , sensor_cols)
#         train_df = train_df.drop(columns=columns_to_drop)
#         #display(train_df)
#         #trainedSets[trainKey] = train_df
#         print("Processing : ", train_df['CYCLE'].count(), " rows")
#         #print("Not in Trained Set")

#         if len(test_df)>0:
#           print("Test data is in memory!")
#         else:
#           print("Loading test data.")
#           testDataSetKeys = testMap.keys()
#           for testKey in testDataSetKeys:
#             test_df = getDataFromCSV(sqlContext, dbFSDir,testMap[testKey], selectedCols)
#             test_df = test_df.append(test_df)
#             break

#           test_df = test_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
#           test_df = add_features(test_df, rolling_win_size , sensor_cols)
#           test_df = test_df.drop(columns=columns_to_drop)

#         if trainKey in valDataSetKeys:
#           print("Loading Validate dataset[",trainKey,"]",valMap[trainKey])
#           val_df = getDataFromCSV(sqlContext, dbFSDir,valMap[trainKey], selectedCols)
#           val_df = val_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
#           val_df = add_features(val_df, rolling_win_size , sensor_cols)
#           val_df = val_df.drop(columns=columns_to_drop)
#         else:
#           print("Key not found in Validation Map, please verify the index file.")
#           #print("Builing model.")
#         printDFPortion(train_df, val_df,"LABEL1")

#         processCode = str(cLoop)+"_R_"+str(countTrainSet)
#         history, old_weights, model = trainModel("LSTM",old_weights,train_df,val_df, test_df ,processCode,tensor_board,model)

#         if len(old_weights)==0:
#           print("Error Empty Weights!!")
#         else:
#           print("Has weights!!")

#         try:
#           lossOptimal, score, resutl, curVLoss = isOptimal(history,countTrainSet,score,curVLoss,nEpoch)
#         except:
#           print("Error in lossOptimal..")
#           pass

#         resultMetric[cLoop] = [processCode] + resutl
#         print(resultMetric)

#         if lossOptimal is False:
#           countTrainSet=countTrainSet+1
#         else:
#           break  

#       else:
#         print("Train and evaluation is satisfactory!")
#         break

# ##run_train_all()

# COMMAND ----------

# resultMetrixLSTM = pd.DataFrame.from_dict(resultMetric, orient='index', columns=['m_code', 'Loss', 'Acc', 'Val_Loss', 'Val_Acc'])
# resultMetrixLSTM = resultMetrixLSTM.sort_values([ 'm_code'])

# print("%.2f%% (+/- %.2f%%)" % (np.mean(resultMetrixLSTM['Acc']), np.std(resultMetrixLSTM['Acc'])))

# resultMetrixLSTM.to_csv("result_BDLSTM.csv")
# fromPathCSV = "file:/databricks/driver/result_BDLSTM.csv"

# try:
#   print("Copying file [",fromPathCSV,"] to Data Lake....")
#   copyData(fromPathCSV, dataLake+"/",False)
# except:
#   print("Error while trying to transfer file "+"file:/databricks/driver/"+fromPathCSV," to ",dataLake+"/")
#   pass

# try:
#   dbutils.fs.cp("file:/tmp/tfb_log_dir/", "dbfs:/mnt/Exploratory/WCLD/12_BDLSTM_RV5/tensorboard_log/",recurse=True)
# except:
#   print("Not found tensorboard file or copying error!")
#   pass

# COMMAND ----------

try:
  print("Copying file tensorboard to Data Lake....")
  copyData("file:/"+tfb_log_dir, tensorboardLogPath ,True)
except:
  print("Error while trying to transfer tensorboard file.")
  pass

# COMMAND ----------

def loadDataSet(valMap):
#   train_df = getDataFromCSV(sqlContext, dbFSDir,trainMap[1], selectedCols,True)
#   train_df = train_df.append(train_df)
#   train_df = train_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
#   train_df = add_features(train_df, rolling_win_size , sensor_cols)
#   train_df = train_df.drop(columns=columns_to_drop)
  print("Loading validate dataset")
  valDataSetKeys = valMap.keys()
  test_df={}
  val_df={}
  min_max_scaler = preprocessing.MinMaxScaler()
  withUpsampling = False

  for vKey in valDataSetKeys:
    val_df_new = getDataFromCSV(sqlContext, dbFSDir,valMap[vKey], selectedCols,withUpsampling)
    val_df_new = val_df_new.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
    val_df_new = add_features(val_df_new, rolling_win_size , sensor_cols)
    val_df_new = val_df_new.drop(columns=columns_to_drop)
    
    if len(val_df)>0:
      val_df = val_df.append(val_df_new)
    else:
      val_df = val_df_new
  
  val_df = genSampleLabel(val_df)
  #val_df = normalizeMaxMinTrain(val_df,min_max_scaler)
  val_df = val_df.sort_values(['EVENT_ID','CYCLE'])
  val_df = val_df.drop_duplicates(['EVENT_ID','CYCLE'], keep='last')

  return val_df

# COMMAND ----------

# print("Loading test dataset")
# testDataSetKeys = testMap.keys()

# for tKey in testDataSetKeys:
#   test_df_new = getDataFromCSV(sqlContext, dbFSDir,testMap[tKey], selectedCols,True)
#   test_df_new = test_df_new.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
#   ##test_df_new = add_features(test_df_new, rolling_win_size , sensor_cols)
#   test_df_new = test_df_new.drop(columns=columns_to_drop)

#   if len(test_df)>0:
#     test_df = test_df.append(test_df_new)
#   else:
#     test_df = test_df_new

# test_df = normalizeMaxMinTrain(test_df,min_max_scaler)

# COMMAND ----------

# def testGen(targetColName, dataFrame):
  
#   ##print(sequence_cols)
#   x_array, y_array , nb_features, nb_out = gen_data_train_val(targetColName, dataFrame,sequence_length, sequence_cols)
#   print("Finish Gen Data Sequence")

# #   val_seq_array, val_label_array, val_nb_features, valNb_out = gen_data_train_val(targetColName, val_df,sequence_length, sequence_cols)
# #   print("Finish Gen Validate Data Sequence")
  
# #   test_seq_array, test_label_array, test_nb_features, testNb_out = gen_data_train_val(targetColName, test_df,sequence_length, sequence_cols)
# #   print("Finish Gen Test Data Sequence")
  
#   return x_array, y_array, nb_features, nb_out
  

# COMMAND ----------

def sampleModelTest():  
  model = Sequential()
  modelMode = "BINARY"

  if modelMode=="MULTI":
    label_array = keras.utils.to_categorical(label_array, num_classes=3, dtype='int32')
    valLabel_array = keras.utils.to_categorical(valLabel_array, num_classes=3, dtype='int32')
    print("after label_array:",valLabel_array[0:5])
    print("valLabel_array", valLabel_array[0:5])
    nb_classes=valLabel_array.shape[1]
    print("nb_classes:",nb_classes)

    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features)))
    model.add(Dropout(v_Dropout))
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2)))
    model.add(Dropout(v_Dropout))
    model.add(Dense(units=nb_classes))
    model.add(Activation('softmax'))
  else:
    nb_out = label_array.shape[1]
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features),merge_mode='concat'))
    model.add(Dropout(v_Dropout))
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2, return_sequences=False))) 
    model.add(Dropout(v_Dropout))
    model.add(Dense(units=nb_out, activation='sigmoid'))

  try:
    model = multi_gpu_model(model,gpus=4)
    print("Training using multiple GPUs..")
  except BaseException as ex:
    print("Training using single GPU or CPU..")
    print('Failed '+ str(ex))
    # Get current system exception
    ex_type, ex_value, ex_traceback = sys.exc_info()
    # Extract unformatter stack traces as tuples
    trace_back = traceback.extract_tb(ex_traceback)
    # Format stacktrace
    stack_trace = list()

    for trace in trace_back:
      stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))

    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" %ex_value)
    print("Stack trace : %s" %stack_trace)

  v_optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

  if modelMode=="MULTI":
    model.compile(loss='categorical_crossentropy', optimizer=v_optimizer, metrics=['categorical_accuracy','accuracy'])
  else:
    model.compile(loss='binary_crossentropy', optimizer=v_optimizer, metrics=['accuracy'])

  print(model.summary())

  history = model.fit(seq_array, label_array,
            batch_size=v_batch_size, epochs=v_maxEpoch, shuffle=False, verbose=2,
            validation_split=0.05,
            callbacks = [tensor_board, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                           keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])

# COMMAND ----------

def evaluation():
  scores_test = model.evaluate(valSeq_array, valLabel_array, verbose=1)
  print("scores_test:",scores_test)
  y_true = valLabel_array
  y_pred = model.predict(valSeq_array)
  categorical_acc = keras.metrics.categorical_accuracy(y_true, y_pred)

  testLabel_array = keras.utils.to_categorical(testLabel_array, num_classes=3, dtype='int32')
  y_test_true = testLabel_array
  y_test_pred = model.predict(testSeq_array)

  #print("y_pred", y_pred) 
  
  labels = np.argmax(y_pred, axis=-1)
  lb = preprocessing.LabelBinarizer()
  label2 = lb.fit_transform(labels)

  print("labels:",labels[0:20])
  print("labels2:",label2[0:20])
  # labels = (y_pred > 0.5).astype(np.int)
  # print("labels:",labels)
  # print("y_true[0:20]", y_true[0:20])
  # print("y_pred[0:20]", y_pred[0:20]) 
  print('Evaluation Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))
  print('Categorical Accurracy:',categorical_acc)

  y_true_non_category = [ np.argmax(t) for t in y_true]
  y_predict_non_category = [ np.argmax(t) for t in y_pred ]

  cm = confusion_matrix(y_target=y_true_non_category, 
                        y_predicted=y_predict_non_category, 
                        binary=False)

      #print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
      #cm = confusion_matrix(y_true, y_pred)
  print(cm)
  fig, ax = plot_confusion_matrix(conf_mat=cm)
  display()

  cm = confusion_matrix(y_target=y_true_non_category, 
                        y_predicted=y_predict_non_category,
                        binary=True, 
                        positive_label=1)
  cm
  fig, ax = plot_confusion_matrix(conf_mat=cm)
  display()
  y_test_true_non_category = [np.argmax(t) for t in y_test_true]
  y_test_predict_non_category = [ np.argmax(t) for t in y_test_pred ]
  cm = confusion_matrix(y_target=y_test_true_non_category, 
                        y_predicted=y_test_predict_non_category, 
                        binary=True)

  #print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
  #cm = confusion_matrix(y_true, y_pred)
  print(cm)

  predicted1 = np.argmax(y_test_pred, axis=-1)
  print(predicted1[0:20])
  predicted2 = np.argmax(y_test_true, axis=1)
  print(predicted2[0:20])
  test_df2 = getDataFromCSV(sqlContext, dbFSDir,testMap[1], selectedCols,True)
  test_df2 = test_df2.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
  testDF = spark.createDataFrame(test_df2)

  testDF.createOrReplaceTempView("TEMP_DATA")
  testDFS = spark.sql("SELECT LABEL2, COUNT(*) FROM TEMP_DATA GROUP BY LABEL2")
  display(testDFS)

  testLabel_array = keras.utils.to_categorical(testLabel_array, num_classes=3, dtype='int32')
  scores_test_df = model.evaluate(testSeq_array, testLabel_array , verbose=1)
  print("scores_test:",scores_test_df)

  #from sklearn.metrics import fmeasure
  fbeta = fbeta_score(y_true_non_category, y_predict_non_category, beta=1,average='macro')
  #fm = fmeasure(y_true_non_category, y_predict_non_category)
  print('fbeta :',fbeta)
  #print('fmeasure :',fm)

# COMMAND ----------

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


#y_true, y_pred = np.round(np.random.rand(100)), np.random.rand(100)
def evaluationTest():
  y_true, y_pred = y_true_non_category, y_predict_non_category
  fbeta_keras = fbeta(K.variable(y_true), K.variable(y_pred)).eval(session=K.get_session())
  fbeta_sklearn = fbeta_score(y_true, np.round(y_pred), beta=2,average='macro')
  print('Scores are {:.3f} (sklearn) and {:.3f} (keras)'.format(fbeta_sklearn, fbeta_keras))

  conf_mat = confusion_matrix(y_target=y_true_non_category, 
                        y_predicted=y_predict_non_category, 
                        binary=True)
  conf_mat

  predicted1 = np.argmax(y_pred, axis=-1)
  print(predicted1[0:20])
  predicted2 = np.argmax(y_pred, axis=1)
  print(predicted2[0:20])

  predicted = np.argmax(y_pred, axis=1)
  report = classification_report(np.argmax(y_true, axis=1), predicted)
  print(report)

# COMMAND ----------

# def evaluationMetrics(valSeq_array, valLabel_array,isBinary,model):
  
#   scores_test = model.evaluate(valSeq_array, valLabel_array, verbose=1)
#   print('Evaluation Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))
#   print("Score: {} ".format(round_down_float(scores_test[0]*100,2)))
#   y_true = valLabel_array
  
#   y_true_label = valLabel_array
#   cm = []
  
#   if (isBinary):
#     try:
#       y_pred_class = model.predict_classes(valSeq_array)
#       #y_true_label = np.argmax(y_true_label, axis=1)
#     except:
#       print("Error CM1")
#       y_pred_prop = model.predict(valSeq_array)
#       y_pred_class = np.argmax(y_pred_prop, axis=1)
#       #y_true_label = np.argmax(y_true_label, axis=1)
#       pass
    
#     print("y_true:", y_true_label[0:20])
#     print("y_pred:", y_pred_class[0:20])
    
#     print('[Exten ML] Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
#     cm = confusion_matrix(y_target=y_true_label, 
#                 y_predicted=y_pred_prop, 
#                 binary=isBinary)
#     print(cm)
    
#     report = classification_report(y_true_label, y_pred_prop)
    
#     print(report)
    
#     y_true_non_category = [ np.argmax(t) for t in y_true_label]
#     y_predict_non_category = [ np.argmax(t) for t in y_pred_prop]
    
#     # compute precision and recall
#     #precision_test =precision_test (y_true_non_category, y_predict_non_category)
    
#     precision_test = precision_score(y_true_label, y_pred_prop , labels=[0,1], pos_label = 1)
#     ##print(" precision_test: ",precision_test)
#     recall_test = recall_score(y_true_label, y_pred_prop , labels=[0,1], pos_label = 1)
#     ##print(" recall_test: ",recall_test)
    
#     if (precision_test + recall_test)>0:
#       f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
#     else:
#       f1_test = 0
      
#     print('Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )
    
#     # Plot in blue color the predicted data and in green color the
#     # actual data to verify visually the accuracy of the model.    
#     return cm, precision_test,recall_test,f1_test, y_true_label, y_predicted
  
#   else:
#     try:
#       #y_pred = model.predict_classes(valSeq_array)
#       y_pred_prop = model.predict(valSeq_array)
#       y_pred_class = np.argmax(y_pred_prop, axis=1)
      
#     except:
#       print("Error CM1")
#       pass
#     try:
#       y_true_label_classes = np.argmax(y_true_label, axis=-1)     
#       try:
        
#         y_true_non_category = [ np.argmax(t) for t in y_true_label]
#         y_predict_non_category = [ np.argmax(t) for t in y_pred_class]
        
#         print('[Exten ML] Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')

#         conf_mat = confusion_matrix(y_target=y_true_label_classes, 
#                           y_predicted=y_pred_class, 
#                           binary=False)
#         cm = conf_mat
#         print(conf_mat)
        
#         report = classification_report(np.argmax(y_true_label, axis=1), y_pred_class)
       
#         print(report)
#       except:
#         print("Error CM1 and classification_report")
#         pass
      
#       try:
        
#         print("+++++Using True Classes[macro]+++++")
#         precision,recall,fscore,support=score(y_true_label_classes , y_pred_class,average='macro')
#         print('Precision : {}%'.format(round_down_float(precision*100,2)))
#         print('Recall    : {}%'.format(round_down_float(recall*100,2)))
#         print('F-score   : {}%'.format(round_down_float(fscore*100,2)))
#         print('Support   : {}'.format(support))
        
#         print("+++++Using True Classes[micro]+++++")
#         precision,recall,fscore,support=score(y_true_label_classes , y_pred_class,average='micro')
#         print('Precision : {}%'.format(round_down_float(precision*100,2)))
#         print('Recall    : {}%'.format(round_down_float(recall*100,2)))
#         print('F-score   : {}%'.format(round_down_float(fscore*100,2)))
#         print('Support   : {}'.format(support))
        
#         print("+++++Using True Classes[weighted]+++++")
#         precision,recall,fscore,support=score(y_true_label_classes , y_pred_class,average='weighted')
#         print('Precision : {}%'.format(round_down_float(precision*100,2)))
#         print('Recall    : {}%'.format(round_down_float(recall*100,2)))
#         print('F-score   : {}%'.format(round_down_float(fscore*100,2)))
#         print('Support   : {}'.format(support))
        
#       except:
#         print("Error in calculating F Score.")
#         pass
#     except:
#       pass
    
#     return cm,precision, recall,fscore, y_true_label_classes , y_pred_class

def evaluationMetrics(valSeq_array, valLabel_array,isBinary,model):
  
  scores_test = model.evaluate(valSeq_array, valLabel_array, verbose=1)
  print('Evaluation Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))
  print("Score: {} ".format(round_down_float(scores_test[0]*100,2)))
  y_true = valLabel_array
  
  y_true_label = valLabel_array
  cm = []
  
  if (isBinary):
    try:
      y_pred_class = model.predict_classes(valSeq_array)
      #y_true_label = np.argmax(y_true_label, axis=1)
    except:
      print("Error CM1")
      y_pred_prop = model.predict(valSeq_array)
      y_pred_class = np.argmax(y_pred_prop, axis=1)
      #y_true_label = np.argmax(y_true_label, axis=1)
      pass
    
    print("y_true:", y_true_label[0:20])
    print("y_pred:", y_pred_class[0:20])
    
    print('[Exten ML] Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
    cm = confusion_matrix(y_target=y_true_label, 
                y_predicted=y_pred_prop, 
                binary=isBinary)
    print(cm)
    
    report = classification_report(y_true_label, y_pred_prop)
    
    print(report)
    
    y_true_non_category = [ np.argmax(t) for t in y_true_label]
    y_predict_non_category = [ np.argmax(t) for t in y_pred_prop]
    
    # compute precision and recall
    #precision_test =precision_test (y_true_non_category, y_predict_non_category)
    
    precision_test = precision_score(y_true_label, y_pred_prop , labels=[0,1], pos_label = 1)
    ##print(" precision_test: ",precision_test)
    recall_test = recall_score(y_true_label, y_pred_prop , labels=[0,1], pos_label = 1)
    ##print(" recall_test: ",recall_test)
    
    if (precision_test + recall_test)>0:
      f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
    else:
      f1_test = 0
      
    print('Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test )
    
    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.    
    return cm, precision_test,recall_test,f1_test, y_true_label, y_predicted
  
  else:
    try:
      #y_pred = model.predict_classes(valSeq_array)
      y_pred_prop = model.predict(valSeq_array)
      y_pred_class = np.argmax(y_pred_prop, axis=1)
      
    except:
      print("Error CM1")
      pass
    try:
      y_true_label_classes = np.argmax(y_true_label, axis=-1)     
      try:
        
        y_true_non_category = [ np.argmax(t) for t in y_true_label]
        y_predict_non_category = [ np.argmax(t) for t in y_pred_class]
        
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
        precision = 0
        recall=0
        fscore=0
        support=0
        
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
        
      except:
        print("Error in calculating F Score.")
        pass
      
    except:
      print("Error in calculating evaluationMetrics ")
      pass
    
    return cm,precision, recall,fscore, y_true_label_classes , y_pred_class

# COMMAND ----------

def plot_history(history, title):
  # list all data in history
  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['categorical_accuracy'])
  plt.plot(history.history['val_categorical_accuracy'])
  plt.title('Accuracy: '+title)
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  display()
  
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Loss: '+title)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig("./plot_history_"+title+".jpg") 
  display()

# COMMAND ----------

def runChart(y_pred_test,y_true_test,model_name,data_set_name):
  fig_verify = plt.figure(figsize=(10, 5))
  plt.plot(y_pred_test, color="blue")
  plt.plot(y_true_test, color="green")
  plt.title('prediction')
  plt.ylabel('value')
  plt.xlabel('row')
  plt.legend(['predicted', 'actual data'], loc='upper left')
  datafile = data_set_name+"_"+model_name+"_"+"_model_verify.png"
  fig_verify.savefig(datafile)

# COMMAND ----------

# try:
#   plot_history(history,"History categorical_accuracy")
# except:
#   pass

# COMMAND ----------

# fbeta = fbeta_score(y_true, y_pred, beta=1)
# fm = fmeasure(y_true, y_pred)

# COMMAND ----------

# ## Stop Tensorboard
# dbutils.tensorboard.stop()

# COMMAND ----------

### End Experiment ######

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

#train_seq_array, train_label_array, nb_features, nb_out = testGen(targetColName, train_df)
targetColName = "LABEL2"
isBinary = False

val_df = loadDataSet(valMap) 
test_df = loadDataSet(testMap)

printDFPortion(val_df, test_df, targetColName)

# COMMAND ----------

#val_seq_array, val_label_arra, nb_features, nb_out = testGen(targetColName, val_df)
val_seq_array, val_label_arra = gen_data_test_val(targetColName, val_df,sequence_length, sequence_cols)
val_label_array = to_categorical(val_label_arra, num_classes=3, dtype='int32')

# COMMAND ----------

cm,precision, recall,fscore, y_true_label_classes , y_pred_class = evaluationMetrics(val_seq_array, val_label_array,isBinary,model)

# COMMAND ----------

print(cm)

# COMMAND ----------

cm = confusion_matrix(y_target=y_true_label_classes, 
                y_predicted=y_pred_class, 
                binary=False)

fig, ax = plot_confusion_matrix(conf_mat=cm)
display()

# COMMAND ----------

try:
  runChart(y_pred_class,y_true_label_classes,"2_R_1_12_BDLSTM_RV6","val")
except:
  cm = confusion_matrix(y_target=y_true_label, 
                y_predicted=y_predicted, 
                binary=False)
  pass
display()

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:mnt/Exploratory/WCLD/12_BDLSTM_RV6/model

# COMMAND ----------

print("\n+++++++++++++\n")
print("===Testing Prediction====")
test_seq_array, test_label_array = gen_data_test_val(targetColName, test_df,sequence_length, sequence_cols)
test_label_array = to_categorical(test_label_array, num_classes=3, dtype='int32')
#test_seq_array, test_label_array, nb_features, nb_out =  testGen(targetColName, test_df)
cm,precision, recall,fscore, y_true_label_classes , y_pred_class = evaluationMetrics(test_seq_array, test_label_array ,isBinary, model)

# COMMAND ----------

print(cmTest)

# COMMAND ----------

cmTest = confusion_matrix(y_target=y_true_label_classes, 
                y_predicted=y_pred_class, 
                binary=True)

fig, ax = plot_confusion_matrix(conf_mat=cmTest)
display()

# COMMAND ----------

cmTest = confusion_matrix(y_target=y_true_label_classes, 
                y_predicted=y_pred_class, 
                binary=False)

fig, ax = plot_confusion_matrix(conf_mat=cmTest)
display()

# COMMAND ----------

fig, ax = plot_confusion_matrix(conf_mat=cm)
display()

# COMMAND ----------

