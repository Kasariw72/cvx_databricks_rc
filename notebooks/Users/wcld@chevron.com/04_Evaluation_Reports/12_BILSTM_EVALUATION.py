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
import math

from itertools import cycle
from sklearn import metrics


import os
# Setting seed for reproducibility
np.random.seed(1234)  
PYTHONHASHSEED = 0
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelBinarizer

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
sparkAppName = "12_BDLSTM_RV4"
# define path to save model
model_path = sparkAppName+".h5"

dirPath = "file:/databricks/driver/"
dataLake = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
mntDatalake = "/mnt/Exploratory/WCLD/"+sparkAppName

tf_at_dl_dir = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
dbFSDir = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV3"
tensorboardLogPath = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName+"/tensorboard_log"

timestamp = datetime.datetime.now().strftime("%Y%m%d%H")

expContext = sparkAppName+"_"+str(timestamp)
tfb_log_dir = tfb_log_dir+expContext
print(expContext)
### Enable Tensorboard and save to the localtion
#dbutils.tensorboard.start(tfb_log_dir)

# COMMAND ----------

def copyData(fromPath, toPath,recurseFlag=True):
  dbutils.fs.cp(fromPath, toPath,recurse=recurseFlag)
  print("Completed copying ["+fromPath+"] to "+toPath)
  
def round_down_float(n, decimals=0):
    multiplier = 10 ** decimals
    return float(math.floor(n * multiplier) / multiplier)

# COMMAND ----------

# print(expContext)
# #dbutils.fs.mkdirs(dirPath+"/"+expContext)
# print(dirPath+expContext)
# dbutils.fs.mkdirs("dbfs:/RC/experiments/"+expContext)
# print("dbfs:/RC/experiments/"+expContext)
# expDir = "dbfs:/RC/experiments/"+expContext

# COMMAND ----------


# sequence_cols = ['RESI_LL_LUBE_OIL_PRESS', 'RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_SPEED',
# 'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',
# 'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP', 'LUBE_OIL_PRESS',
# 'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP',
# 'THROW_1_SUC_PRESS', 'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS',
# 'THROW_2_DISC_TEMP', 'THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS',
# 'THROW_3_DISC_TEMP', 'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP',
# 'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP',
# 'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP', 'CYL_6_TEMP', 
# 'CYL_7_TEMP', 'CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP', 'LUBE_OIL_PRESS_ENGINE', 'SPEED', 'VIBRA_ENGINE']

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

sequence_cols = sensor_cols 

# COMMAND ----------

# MAGIC %md ## Data Ingestion
# MAGIC Training Data

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

def convertStr2Int(pySparkDF, intColumns,doubleColumns):
    for colName in intColumns:
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(IntegerType()))    
      pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(0))
      
    for colName in doubleColumns:
      pySparkDF = pySparkDF.withColumn(colName,pySparkDF[colName].cast(DoubleType()))
      pySparkDF = pySparkDF.withColumn(colName, when(pySparkDF[colName].isNotNull(),pySparkDF[colName]).otherwise(0))
    return pySparkDF

intCols = ['LABEL1','LABEL2','EVENT_ID','CYCLE']

def getFilePathInDBFS(dbFSDir,eventId):
  dfDirDBFS = dbutils.fs.ls(dbFSDir)
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

def getDataFromCSV(sqlContext, dbFSDir,eventIds, selectedCols):
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
      loadDF = loadDF.selectExpr(selectedCols)
      
      if df != "":
        df = df.union(loadDF)
      else:
        df = loadDF
        
  df.cache()
  
  print("Spark finishs caching dataframe!")
  selectColDF = df.selectExpr(selectedCols)
  selectColDF = convertStr2Int(selectColDF, intCols, sensor_cols)
  
  #compute dataframe using sql command via string
  selectColDF.createOrReplaceTempView("RC_DATA_TMP")
  #filter out the data during RC S/D
  selectColDF = spark.sql("SELECT * FROM RC_DATA_TMP ORDER BY YEAR, EVENT_ID, CYCLE")
  #spDataFrame = df.drop(*columns_to_drop)
  selectColDF = selectColDF.toPandas()
  #selectDataCols = selectDataCols.sort_values(['YEAR','MONTH','DAY','EVENT_ID','CYCLE'])
  return selectColDF

# COMMAND ----------

## Loading Index Data Table
indexFilePath = "dbfs:/mnt/Exploratory/WCLD/dataset/EVENT_ID_DATA_SETS_FULL_V5.csv"
df2 = (sqlContext.read.option("header","true").option("inferSchema", "true").csv(indexFilePath))
# If the path don't have file:/// -> it will call hdfs instead of local file system
df2.cache()
#print("Spark finishs caching:"+indexFilePath)

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

# MAGIC %md ## Deep Learining - LSTM for Binary Classification Objective

# COMMAND ----------

# pick a large window size of 120 cycles
# Target 1 days using 120 time steps

sequence_length = 100
input_length = 100

sensor_av_cols = ["AVG_" + nm for nm in sensor_cols]
sensor_sd_cols = ["SD_" + nm for nm in sensor_cols]

sequence_cols = sequence_cols + sensor_av_cols + sensor_sd_cols
#sequence_cols = sequence_cols.append(sensor_sd_cols)

#print(sequence_cols)
columns_to_drop = ['CODE','YEAR']
dbFSDir = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV3"

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
  # sensor_cols = ['s' + str(i) for i in range(1,22)]
  # sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
  # sequence_cols.extend(sensor_cols)

  ##print("sequence_length",sequence_length)
  ##print("sequence_cols",sequence_cols)
  
  # generator for the sequences
  seq_gen = (list(gen_sequence(train_df[train_df['EVENT_ID']==id], sequence_length, sequence_cols)) 
             for id in train_df['EVENT_ID'].unique())

  # generate sequences and convert to numpy array
  seq_array = np.concatenate(list(seq_gen)).astype(np.float64)
  seq_array.shape
  
  # generate labels
#   label_gen = [gen_labels(train_df[train_df['EVENT_ID']==id], sequence_length, train_df['LABEL1']) 
#                for id in train_df['EVENT_ID'].unique()]
  
#   label_array = np.concatenate(label_gen).astype(np.float64)
#   label_array.shape
  
  # generate labels
  label_gen = [gen_labels(train_df[train_df['EVENT_ID']==id], sequence_length, [targetColName]) 
             for id in train_df['EVENT_ID'].unique()]
  label_array = np.concatenate(label_gen).astype(np.float64)
  label_array.shape

  # Next, we build a deep network. 
  # The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 
  # Dropout is also applied after each LSTM layer to control overfitting. 
  # Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem.
  # build the network
  
  nb_features = seq_array.shape[2]
  #print(">> nb_features:",nb_features)
  nb_out = label_array.shape[1]
  #print(label_array)
  #print("1>> nb_out:",nb_out)
  
#   if targetColName=="LABEL2":
#     nb_out = label_array.shape[1]
#     print("2>> nb_out:",nb_features)
#   else:
#     nb_out = label_array.shape[1]
#     print("1>> nb_out:",nb_features)
    
  return seq_array, label_array, nb_features, nb_out

def trainModel(modelCode, old_weights,train_df,val_df,test_df,iterNum,tensor_board,model):
  modelNameList = ["buildBinaryClassModel","buildMultipleClassModel","buildMultiAttentionModel"]
  modelCodeMap = {"LSTM":"buildBinaryClassModel", "BDLSTM":"buildMultipleClassModel","BDLSTM_ATTEN":"buildMultiAttentionModel"}
  print("Start Train Model ",iterNum)
  targetColName = "LABEL1"
  
  if modelCodeMap[modelCode] == "buildMultipleClassModel":
    targetColName = "LABEL2"
  elif modelCodeMap[modelCode]  == "buildMultiAttentionModel":
    targetColName = "LABEL2"
  else:
    modelCodeMap[modelCode]  = "LABEL1"
  
  seq_array, label_array, nb_features, nb_out = gen_data_train_val(targetColName, train_df,sequence_length, sequence_cols)
  print("Finish Gen Train Data Sequence")
  
  valSeq_array, valLabel_array, valNb_features, valNb_out = gen_data_train_val(targetColName, val_df,sequence_length, sequence_cols)
  print("Finish Gen Validate Data Sequence")
 #Hyperparameters
  
  v_batch_size = 200
  v_validation_split = 0.05
  v_verbose = 2
  #verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
  
  v_LSTMUnitLayer1 = 100
  v_LSTMUnitLayer2 = 60
  v_Dropout = 0.2
  v_maxEpoch = 1
  
  ## Run Single GPU
  try:
    model = multi_gpu_model(model,gpus=4)
    print("Training using multiple GPUs..")
  except:
    print("Training using single GPU or CPU..")
    pass
  
  try:
    if old_weights!="":
      model.set_weights(old_weights)
      print("Reset weights successfully.")      
  except:
    print("Failed reset weights.")
    pass
  
  if modelCodeMap[modelCode]==modelNameList[0]:
    
    model, history = buildBinaryClassModel(modelCode,
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
                                                     valLabel_array)
  elif modelCodeMap[modelCode]==modelNameList[1]:
    model, history = buildMultipleClassModel(modelCode,
                                                       old_weights,
                                                       v_LSTMUnitLayer1,
                                                       v_LSTMUnitLayer2,
                                                       sequence_length,
                                                       nb_features,
                                                       nb_out,v_Dropout,
                                                       v_maxEpoch,
                                                       v_batch_size,
                                                       seq_array, 
                                                       label_array,
                                                       valSeq_array, 
                                                       valLabel_array)
  elif modelCodeMap[modelCode]==modelNameList[2]:
    model, history = buildMultipleClassAttentionModel(modelCode,
                                                       old_weights,
                                                       v_LSTMUnitLayer1,
                                                       v_LSTMUnitLayer2,
                                                       sequence_length,
                                                       nb_features,
                                                       nb_out,v_Dropout,
                                                       v_maxEpoch,
                                                       v_batch_size,
                                                       seq_array, 
                                                       label_array,
                                                       valSeq_array, 
                                                       valLabel_array)
  else:
    print("Error, please verify and input model name!")
    
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
    
#     try:
#       print("Copying file tensorboard to Data Lake....")
#       copyData("file:/"+tfb_log_dir, tensorboardLogPath ,True)
#     except:
#       print("Error while trying to transfer tensorboard file.")
#       pass
    
    print("Model Saved >> ",currentModelPath)
    ##evaluationTest(iterNum, model,seq_array,history,label_array,iterNum,currentModelPath)
  except:
    print("Error Saving Model",currentModelPath)
    pass
  
#Evaluate Model
#   try:
#     evaluationValidationSet(iterNum, test_df,currentModelPath)
#   except:
#     print(" Error evaluationValidationSet ")
#     pass
  return history, old_weights , model

# COMMAND ----------

def buildBinaryClassModel(algoName,old_weights,v_LSTMUnitLayer1,v_LSTMUnitLayer2,sequence_length,nb_features,nb_out,v_Dropout,v_maxEpoch,v_batch_size,seq_array, label_array,valSeq_array, valLabel_array):
  if old_weights=="":
    print("Created New Model : "+algoName," Algo Mode :",algoName)
    if modelName=="BDLSTM":
      model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features),merge_mode='concat'))
      model.add(Dropout(v_Dropout))
      model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2, return_sequences=False))) 
      model.add(Dropout(v_Dropout))
      model.add(Dense(units=nb_out, activation='sigmoid'))
    else:
      print("Created New Model : "+modelName)
      model.add(LSTM(input_shape=(sequence_length, nb_features), units=v_LSTMUnitLayer1, return_sequences=True)) 
      model.add(Dropout(v_Dropout))
      model.add(LSTM(units=v_LSTMUnitLayer2, return_sequences=False)) 
      model.add(Dropout(v_Dropout))
      model.add(Dense(units=nb_out, activation='sigmoid'))
  else:
    print("Model Already Constructed.")
    
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  #fit the network
  history = model.fit(seq_array, label_array, epochs=v_maxEpoch, batch_size=v_batch_size, validation_data=(valLabel_array, valSeq_array), verbose=2,
            callbacks = [tensor_board, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                         keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])  
  return model, history

# COMMAND ----------

def buildMultipleClassModel(algoName,old_weights,v_LSTMUnitLayer1,v_LSTMUnitLayer2,sequence_length,nb_features,nb_out,v_Dropout,v_maxEpoch,v_batch_size,seq_array, label_array,valSeq_array, valLabel_array):
  
  nb_classes=3
  
  # y_train = np_utils.to_categorical(y_train, nb_classes)
  # y_test = np_utils.to_categorical(y_test, nb_classes)
  #print("old_weights:",len(old_weights))
  #print("nb_out:",nb_out)
  
  if len(old_weights)==0:
    
    print("Creating Multiple Classification using Bidirectional LSTM : "+algoName)
    print("Created New Model : "+algoName," Algo Mode :",algoName)
    
    if algoName=="BDLSTM":
      #model.add(Embedding(sequence_length, nb_features, input_length=input_length))
      model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features), merge_mode='concat'))
      print("Created Bidirectional 1")
      model.add(Dropout(v_Dropout))
      model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2, return_sequences=True)))
      print("Created Bidirectional 2")
      model.add(Dropout(v_Dropout))
#       model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2,return_sequences=True)))
#       print("Created Bidirectional 3")
#       model.add(Dropout(v_Dropout))
      model.add(Flatten())
      model.add(Dense(units=nb_classes, activation='softmax'))
    else:
      print("Creating Multiple Classification using LSTM : "+algoName)
      model.add(LSTM(batch_input_shape=(v_batch_size, sequence_length, nb_features), units=v_LSTMUnitLayer1, return_sequences=True))
      model.add(Dropout(v_Dropout))
  #   model.add(LSTM(units=v_LSTMUnitLayer2, return_sequences=True, stateful=True))
  #   model.add(Dropout(v_Dropout))
      model.add(LSTM(units=v_LSTMUnitLayer2, stateful=True))
      model.add(Dropout(v_Dropout))
      model.add(Flatten())
      model.add(Dense(units=nb_classes, activation='softmax'))
  else:
    print("Model Already Constructed.")
    
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  
  label_array = keras.utils.to_categorical(label_array, num_classes=3, dtype='int32')
  #print(valLabel_array)
  valLabel_array = keras.utils.to_categorical(valLabel_array, num_classes=3, dtype='int32')
  
  history = model.fit(seq_array, label_array,
          batch_size=v_batch_size, epochs=v_maxEpoch, shuffle=False, verbose=2,
          validation_data=(valSeq_array, valLabel_array),
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
  
  #fit the network
#   history = model.fit(seq_array, label_array, epochs=v_maxEpoch, shuffle=False, batch_size=v_batch_size, validation_data=(valSeq_array, valLabel_array), verbose=2,
#             callbacks = [tensor_board, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
#                          keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
  
#   history = model.fit(seq_array, label_array,
#           batch_size=v_batch_size, epochs=v_maxEpoch, shuffle=False, verbose=2,
#           validation_data=(valSeq_array, valLabel_array),
#            callbacks = [tensor_board, keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
#                          keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])
  
  return model, history

# COMMAND ----------

# MAGIC %md ## Model Evaluation on Test set

# COMMAND ----------

def genLabelArray(train_df):
  # pick the feature columns 
  # sensor_cols = ['s' + str(i) for i in range(1,22)]
  # sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
  # sequence_cols.extend(sensor_cols)
  
  # generator for the sequences
  seq_gen = (list(gen_sequence(train_df[train_df['EVENT_ID']==id], sequence_length, sequence_cols)) 
             for id in train_df['EVENT_ID'].unique())

  # generate sequences and convert to numpy array
  seq_array = np.concatenate(list(seq_gen)).astype(np.float64)
  seq_array.shape

  # generate labels
  label_gen = [gen_labels(train_df[train_df['EVENT_ID']==id], sequence_length, ['LABEL2']) 
               for id in train_df['EVENT_ID'].unique()]
  label_array = np.concatenate(label_gen).astype(np.float64)
  label_array.shape
  return label_array

def genSeqArray(train_df):
    # pick the feature columns 
  # sensor_cols = ['s' + str(i) for i in range(1,22)]
  # sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
  # sequence_cols.extend(sensor_cols)

  # generator for the sequences
  seq_gen = (list(gen_sequence(train_df[train_df['EVENT_ID']==id], sequence_length, sequence_cols)) 
             for id in train_df['EVENT_ID'].unique())

  # generate sequences and convert to numpy array
  seq_array = np.concatenate(list(seq_gen)).astype(np.float64)
  seq_array.shape
  return seq_array

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir output

# COMMAND ----------

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # Run classifier, using a model that is too regularized (C too low) to see
# # the impact on the results
# classifier = svm.SVC(kernel='linear', C=0.01)
# y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# display()

# COMMAND ----------

chartPath = "charts"
dbutils.fs.mkdirs("file:/databricks/driver/"+chartPath)

# COMMAND ----------

# MAGIC %md ## Model Evaluation on Validation set

# COMMAND ----------

#Training Sets
#trainMap,totalEvents = loadDataSet("TRAIN")
#print("Total Events in Traing Dataset : ",str(totalEvents))
#print(trainMap)
# trainMap = {1: [2091, 2101], 
#  2: [712, 714], 
#  3: [924, 950]}

#Validation Sets
# valMap , totalEvents = loadDataSet("VALI")
# print("Total Events in Validation Dataset : ",str(totalEvents))
# print(valMap)
# #valMap = {1: [2124, 642]}

##Test Sets:
testMap,totalEvents = loadDataSet("TEST")
print("Total Events in Test Dataset : ",str(totalEvents))
#testMap = {1: [2124, 642]}
#print(testMap)

# COMMAND ----------

model = Sequential()

# COMMAND ----------

# model = Sequential()
# lastModelPath = "91_R_25_12_BDLSTM_RV3.h5"
# old_weights = ""

# if os.path.isfile(lastModelPath):
#   ##estimator = load_model(modelPath)
#   model = load_model(lastModelPath)
#   old_weights = model.get_weights()
#   print("Model Loaded!",lastModelPath)
# else:
#   print("Model File Not Found!",lastModelPath)

# COMMAND ----------

trainedSets = {}
validatedSets = {}
test_df={}
val_df={}
resultMetric = {}
rolling_win_size = 30

#testMap = valMap 
print("Loading test data.")
testDataSetKeys = testMap.keys()

# for testKey in testDataSetKeys:
#   test_new = getDataFromCSV(sqlContext, dbFSDir,testMap[testKey], selectedCols)
#   test_new = test_new.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
#   test_new = add_features(test_new, rolling_win_size , sensor_cols)
#   test_df = test_df.drop(columns=columns_to_drop)
  
#   seq_array, label_array, valNb_features, valNb_out = gen_data_train_val("LABEL2", test_df,sequence_length, sequence_cols)
#   label_array = keras.utils.to_categorical(label_array, num_classes=3, dtype='int32')
#   print("valNb_features:",valNb_features)
  
# #   if len(test_df)>0:
# #     test_df = test_df.append(test_new)
# #   else:
# #     test_df = test_new



# COMMAND ----------

import pandas as pd

modelMap = {}
modelList = []

for line in dbutils.fs.ls("file:/databricks/driver/models"):
  #print(line[1])
  modelList = []
  modelWords = line[1].split('_')
  modelList.insert(0,int(modelWords[0]))
  modelList.insert(1,int(modelWords[2]))
  modelList.insert(2,'models/'+line[1])
  modelList.insert(3,modelWords[4])
  modelList.insert(4,line[1])
  modelMap[modelWords[0]] = modelList
  
#print(modelMap)
modelMapPD = pd.DataFrame.from_dict(modelMap, orient='index', columns=['Epoch', 'SetN', 'ModelPath','MName','FileName'])
modelMapPD = modelMapPD.sort_values([ 'Epoch'])
display(modelMapPD)

# COMMAND ----------

print(modelMapPD['ModelPath'])

# COMMAND ----------

# modelNum = []
# maxModel = 283
# startModel = 271

# i=startModel
# for i<=maxModel:
#   modelNum.insert(i,startModel)
#   print(startModel)
#   i = i+1

# COMMAND ----------

def multiclass_metrics(model, y_test, y_pred, y_score, print_out=True, plot_out=True):
    
    """Calculate main multiclass classifcation metrics, plot AUC ROC and Precision-Recall curves.
    
    Args:
        model (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        y_score (series): Contains the predicted scores
        print_out (bool): Print the classification metrics and thresholds values
        plot_out (bool): Plot AUC ROC, Precision-Recall, and Threshold curves
        
    Returns:
        dataframe: The combined metrics in single dataframe
        dict: ROC thresholds
        dict: Precision-Recall thresholds
        Plot: AUC ROC
        plot: Precision-Recall
  
    
    """
    multiclass_metrics = {
                            'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                            'macro F1' : metrics.f1_score(y_test, y_pred, average='macro'),
                            'micro F1' : metrics.f1_score(y_test, y_pred, average='micro'),
                            'macro Precision' : metrics.precision_score(y_test, y_pred,  average='macro'),
                            'micro Precision' : metrics.precision_score(y_test, y_pred,  average='micro'),
                            'macro Recall' : metrics.recall_score(y_test, y_pred,  average='macro'),
                            'micro Recall' : metrics.recall_score(y_test, y_pred,  average='micro'),
                            'macro ROC AUC' : metrics.roc_auc_score(y_test, y_score, average='macro'),
                            'micro ROC AUC' : metrics.roc_auc_score(y_test, y_score, average='micro')
                        }
    
    df_metrics = pd.DataFrame.from_dict(multiclass_metrics, orient='index')
    df_metrics.columns = [model]

   
    #n_classes = y_train.shape[1]
    
    n_classes =3
    fpr = dict()
    tpr = dict()
    thresh_roc = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        #fpr[i], tpr[i], thresh_roc[i] = metrics.roc_curve(y_test[:, i], y_score[i][:,0])
        fpr[i], tpr[i], thresh_roc[i] = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], thresh_roc["micro"] = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    

    roc_thresh = {
                    'Threshold' : thresh_roc,
                    'TPR' : tpr,
                    'FPR' : fpr,
                    'AUC' : roc_auc
                 }
    
    df_roc_thresh = pd.DataFrame.from_dict(roc_thresh)
    df_roc_thresh['Model'] = model
    df_roc_thresh['Class'] = df_roc_thresh.index
    
    
    
    precision = dict()
    recall = dict()
    thresh_prc = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        #precision[i], recall[i], thresh_prc[i] = metrics.precision_recall_curve(y_test[:, i], y_score[i][:,0])
        precision[i], recall[i], thresh_prc[i] = metrics.precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = metrics.average_precision_score(y_test[:, i], y_score[:, i])

    precision["micro"], recall["micro"], thresh_prc["micro"] = metrics.precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_test, y_score, average="micro")
    
    prc_thresh = {
                    'Threshold' : thresh_prc,
                    'Precision' : precision,
                    'Recall' : recall,
                    'Avg Precision' : average_precision
                 }

    df_prc_thresh = pd.DataFrame.from_dict(prc_thresh)
    df_prc_thresh['Model'] = model
    df_prc_thresh['Class'] = df_prc_thresh.index    
    
    y_test_orig = lb.inverse_transform(y_test)
    y_pred_orig = lb.inverse_transform(y_pred)
    
    if print_out:
        print('-----------------------------------------------------------')
        print(model, '\n')
        print('Confusion Matrix:')
        print(metrics.confusion_matrix(y_test_orig, y_pred_orig))
        print('\nClassification Report:')
        print(metrics.classification_report(y_test_orig, y_pred_orig))
        print('\nMetrics:')
        print(df_metrics)

    if plot_out:
        
        colors = cycle(['red', 'green', 'blue'])
        
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False )
        fig.set_size_inches(12,6)
        
        for i, color in zip(range(n_classes), colors):
            ax1.plot(fpr[i], tpr[i], color=color, lw=1, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        
        ax1.plot(fpr["micro"], tpr["micro"], color='deeppink', label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]), linestyle=':', linewidth=4)
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([-0.05, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right", fontsize='small')
        
        
        for i, color in zip(range(n_classes), colors):
            ax2.plot(recall[i], precision[i], color=color, lw=1, label='Precision-recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))
            
        ax2.plot(recall["micro"], precision["micro"], color='deeppink', lw=4, linestyle=':', label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc="lower left", fontsize='small')
    
    return df_metrics, df_prc_thresh, df_roc_thresh

# COMMAND ----------

def evaluationValidationSet(iterNum,model,test_df,modelPath,modelName,pathImg):
  accPercent = 0
  precision_test=0 
  recall_test=0 
  f1_test=0
  
  #We pick the last sequence for each id in the test data
  seq_array_test_last = [test_df[test_df['EVENT_ID']==id][sequence_cols].values[-sequence_length:] 
                       for id in test_df['EVENT_ID'].unique() if len(test_df[test_df['EVENT_ID']==id]) >= sequence_length]

  seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float64)
#   print("===seq_array_test_last====")
#   print(seq_array_test_last)
#   print("==========================")
  #seq_array_test_last = keras.utils.to_categorical(seq_array_test_last, num_classes=3, dtype='int32')
  # generate sequences and convert to numpy array
  #seq_array_test_last = np.concatenate(list(seq_array_test_last)).astype(np.float64)
  seq_array_test_last.shape
  
  y_mask = [len(test_df[test_df['EVENT_ID']==id]) >= sequence_length for id in test_df['EVENT_ID'].unique()]

  label_array_test_last = test_df.groupby('EVENT_ID')['LABEL2'].nth(-1)[y_mask].values
  label_array_test_last = keras.utils.to_categorical(label_array_test_last, num_classes=3, dtype='int32')
  label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],3).astype(np.float32)
  
  
#   print("Loading Model Path : ",modelPath)
#   # if best iteration's model was saved then load and use it
#   if os.path.isfile(modelPath):
#       estimator = load_model(modelPath)
#       print("Model Loaded!")
#   # test metrics
  
  scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=1)
  
  print("[",iterNum,"]",'Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))

  accPercent = round_down_float(scores_test[1]*100,2)
  class_names=[0,1,2]
  
  # make predictions and compute confusion matrix
  try:
    
    #y_pred_test = estimator.predict_classes(seq_array_test_last)
    
    predict = model.predict(seq_array_test_last)
    predict=np.argmax(predict,axis=1)
    #predict=np.argmax(predict,label)
    y_pred_test=predict
    print("Success in predict_classes !")
#     print(y_pred_test)
  except:
    print("Error predict_classes !")
    predict = model.predict(seq_array_test_last)
    predict=np.argmax(predict,axis=1)
    y_pred_test=predict
    #predict=np.argmax(predict,label)
    
    pass
   
  y_true_test = label_array_test_last
  test_set = pd.DataFrame(y_pred_test)

  test_set.to_csv(modelName+"_binary_submit_test.csv", index = None)
  
#   print("+++++[y_pred_test]++++++")
#   print(y_pred_test)  
#   print("+++++y_true_test++++++")
#   print(y_true_test)
#   print("+++++++++++")
  
  num_classes = 3

  # from lable to categorial
  y_prediction = y_pred_test
  
  y_categorial = keras.utils.to_categorical(y_prediction, num_classes)
  # from categorial to lable indexing
  y_pred = y_categorial.argmax(1)


  try:
    print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
    t=1
    
    y_test_non_category = [ np.argmax(t) for t in y_true_test ]
    y_predict_non_category = [ np.argmax(t) for t in y_pred_test ]
    
    print("y_test_non_category:")
    print(y_test_non_category)
    print("y_predict_non_category:")
    print(y_predict_non_category)
    #from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    y_pred_test =y_predict_non_category
    y_true_test = y_test_non_category
    
    print(conf_mat)
    
#     matrix = metrics.confusion_matrix(y_true_test, y_pred)
    
#     #cm = confusion_matrix(y_true_test, y_pred_test)
#     print(matrix)
    
#     np.set_printoptions(precision=2)
#     # Plot non-normalized confusion matrix
#     plot_confusion_matrix(y_true_test.argmax(axis=1), y_pred_test.argmax(axis=1), classes=class_names,
#                         title='Confusion matrix, without normalization')
  except:
    print("Error in creating Confusion matrix!")
  
      
#   modelName = 'Bidirectional LSTM'
#   clf_dtrb = model
#   gs_params = {}
#   gs_score = 'roc_auc'
#   y_test = y_true_test
#   y_pred_dtrb =y_pred_test
#   y_score_dtrb = scores_test
#   metrics_dtrb, prc_dtrb, roc_dtrb = multiclass_metrics(model, y_test, y_pred_dtrb, y_score_dtrb, print_out=True, plot_out=True)
    
#   # compute precision and recall
#   try:
#     precision_test = precision_score(y_true_test, y_pred_test)
#     #precision_test = metrics.precision_score(y_true_test, y_pred,  average='macro')
    
#   except:
#     #precision_test = metrics.precision_score(y_true_test, y_pred_test,  average='macro')
#     print("Compute precision error!")
#     pass
  
#   try:
#     recall_test = recall_score(y_true_test, y_pred_test)
#     f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
#     print('Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test)
#   except:
#     print("Cal recall_score error")
#     #recall_test = metrics.recall_score(y_true_test, y_pred_test,  average='macro')
#     #f1_test = metrics.f1_score(y_true_test, y_pred_test,  average='macro')
#     #print('Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test)
#     pass
  
#   fig_verify = plt.figure(figsize=(10, 5))
#   plt.plot(y_pred_test, color="blue")
#   plt.plot(y_true_test, color="green")
#   plt.title('prediction')
#   plt.ylabel('value')
#   plt.xlabel('row')
#   plt.legend(['predicted', 'actual data'], loc='upper left')
  
#   plt.suptitle("Remote Compressure Failure Prediction Evaluation Result "+modelName)
  
#   datafile =pathImg + "/" +modelName+"_model_verify.png"
#   print(datafile)
#   fig_verify.savefig(datafile)

  return accPercent, precision_test, recall_test, f1_test

# COMMAND ----------

def evaluationTest(epochN, model,seq_array, label_array,model_path,modelName):
  
  print("Start Model Evaluation ",epochN)
  if model=="":
    if os.path.isfile(model_path):
      model = load_model(model_path)
      print("Model loaded!")
    else:
      print("Find not found model path:"+model_path)
  
  # metrics
  scores = model.evaluate(seq_array, label_array, verbose=2, batch_size=200)
  print("[",epochN,"]",'Accurracy: {}'.format(scores[1]))
  accPercent = round_down_float(scores[1]*100,2)
  precision = 0
  recall=0
  f1_test =0 
  
  try:
  # make predictions and compute confusion matrix
    y_pred_test = model.predict_classes(seq_array,verbose=2, batch_size=200)
    y_true_test = label_array
  except:
    y_pred_test = model.predict(seq_array)
    y_pred_test=np.argmax(y_pred,axis=1)
    y_true_test=label_array
  
  #test_set = pd.DataFrame(y_pred)
 # test_set.to_csv("output\binary_submit_train"+modelName+".csv", index = None)
  
  #print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
  
  #cm = confusion_matrix(y_true, y_pred)
  #print(cm)
  
  y_test_non_category = [ np.argmax(t) for t in y_true_test ]
  y_predict_non_category = [ np.argmax(t) for t in y_pred_test ]
  conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
  
  y_pred =y_predict_non_category
  y_true = y_test_non_category
    
  print(conf_mat)
    
  #compute precision and recall
  try:
    # compute precision and recall
    precision = precision_score(y_true, y_pred)
  except:
    print("Compute precision error!")
    pass
  
  try:
    recall = recall_score(y_true, y_pred)
    print('precision = ', precision, '\n', 'recall = ', recall)
    
    f1_test = 2 * (precision * recall) / (precision + recall)
    print('Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test)
    
  except:
    print("cal recall_score error")
    pass
  
  return accPercent, precision, recall, f1_test

# COMMAND ----------

# seq_array, label_array, valNb_features, valNb_out = gen_data_train_val("LABEL2", test_df,sequence_length, sequence_cols)
# label_array = keras.utils.to_categorical(label_array, num_classes=3, dtype='int32')
# print("valNb_features:",valNb_features)

# COMMAND ----------

import warnings
warnings.filterwarnings('ignore') 
model = Sequential()
iterNum=1
modelResultDicLSTM = {}

# label_array = genLabelArray(test_df)
# seq_array = genSeqArray(test_df)
# modelMapSample = modelMapPD[ modelMapPD['Epoch']>= 221]
# modelMapPD = modelMapSample

modelMapKey = modelMap.keys()
modelResultDicBDLSTM = {}
scoreMap = {}

for modelKey in modelMapKey:
  modelSet = modelMap[modelKey]
  
  modelPath = modelSet[2]
  modelFullName = modelSet[4]
  epoch = modelSet[0]
  setN = modelSet[1]
  if int(epoch)>=271:
    #print(modelSet)
    if os.path.isfile(modelPath):
      ##estimator = load_model(modelPath)
      model = load_model(modelPath)
      print("Model Loaded!",modelPath)
      scores = model.evaluate(seq_array, label_array, verbose=2, batch_size=200)
      print("[",epoch,"]",'Accurracy: {}'.format(scores[1]))
      accPercent = round_down_float(scores[1]*100,2)
      scoreMap[epoch] = [epoch,setN,modelFullName,accPercent]
      print(scoreMap)
    else:
      print("Model File Not Found!",modelPath)


    #print("**** Model ["+modelPath+"]")
    #print("=======Test metrics====")
    accPercent, precision, recall, f1_test = evaluationTest(epoch, model,seq_array,label_array,model_path, modelFullName)
    #y_train = df_train['label_mcc']
    #y_test = test_df['LABEL2']

  #   lb = LabelBinarizer()
  #   #y_train = lb.fit_transform(y_train)
  #   y_test = lb.transform(y_test)
  #   print("LabelBinarizer:",y_test)
  #   print(y_test)

  #   accPercent, precision, recall, f1_test = evaluationValidationSet(epoch,model,test_df,modelPath,modelFullName,chartPath)
  #   parameters = [epoch, setN, modelPath,'BDLSTM', modelFullName , accPercent, precision*100, recall*100, f1_test*100]

    #accPercent, precision, recall, f1_test = evaluationTest(epoch, model,seq_array, label_array,modelPath,modelFullName)
#     parameters = [epoch, setN, modelPath,'BDLSTM', modelFullName , accPercent, precision*100, recall*100, f1_test*100]
#     #accPercent, precision, recall_test, f1_test = evaluationValidationSet(iterNum,test_df,label_array,modelPath)
#     modelResultDicBDLSTM[epoch] = parameters
    #break

# COMMAND ----------

#testMap = valMap 
print("Loading test data.")
testDataSetKeys = testMap.keys()
results = {}
precision = 0
recall = 0
f1_test = 0
accPercent=0

for testKey in testDataSetKeys:
  test_new = getDataFromCSV(sqlContext, dbFSDir,testMap[testKey], selectedCols)
  test_new = test_new.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
  test_new = add_features(test_new, rolling_win_size , sensor_cols)
  test_df = test_new.drop(columns=columns_to_drop)
  
  seq_array, label_array, valNb_features, valNb_out = gen_data_train_val("LABEL2", test_df,sequence_length, sequence_cols)
  label_array = keras.utils.to_categorical(label_array, num_classes=3, dtype='int32')
  print("valNb_features:",valNb_features)
  accPercent, precision, recall, f1_test = evaluationTest(epoch, model,seq_array,label_array,"models/289_R_25_12_BDLSTM_RV4.h5", "289_R_25_12_BDLSTM_RV4.h5")
  results[testKey]= [testKey,accPercent, precision, recall, f1_test]
  
print(results)

# COMMAND ----------

testKey = 40

test_new = getDataFromCSV(sqlContext, dbFSDir,testMap[testKey], selectedCols)
test_new = test_new.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
test_new = add_features(test_new, rolling_win_size , sensor_cols)
test_df = test_new.drop(columns=columns_to_drop)
  
seq_array, label_array, valNb_features, valNb_out = gen_data_train_val("LABEL2", test_df,sequence_length, sequence_cols)
label_array = keras.utils.to_categorical(label_array, num_classes=3, dtype='int32')
print("valNb_features:",valNb_features)
accPercent, precision, recall, f1_test = evaluationTest(epoch, model,seq_array,label_array,"models/289_R_25_12_BDLSTM_RV4.h5", "289_R_25_12_BDLSTM_RV4.h5")

# COMMAND ----------

try:
  # make predictions and compute confusion matrix
  y_pred_test = model.predict_classes(seq_array,verbose=1, batch_size=200)
  y_true_test = label_array
except:
  y_pred_test = model.predict(seq_array)
  y_pred_test=np.argmax(y_pred,axis=1)
  y_true_test=label_array
  
#test_set = pd.DataFrame(y_pred)
#test_set.to_csv("output\binary_submit_train"+modelName+".csv", index = None)
  
#print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
#cm = confusion_matrix(y_true, y_pred)
#print(cm)  
y_test_non_category = [ np.argmax(t) for t in y_true_test ]
y_predict_non_category = [ np.argmax(t) for t in y_pred_test ]
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)

y_pred =y_predict_non_category
y_true = y_test_non_category

print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
cm = confusion_matrix(y_true, y_pred)
print(cm)

# COMMAND ----------

# #compute precision and recall
# p = precision_score(y_true_test , y_pred_test , average='macro')

try:
  # compute precision and recall
  precision = precision_score(y_true_test, y_pred_test)
except:
  print("Compute precision error!")
  pass
  
try:
  recall = recall_score(y_true, y_pred)
  print('precision = ', precision, '\n', 'recall = ', recall)
    
  f1_test = 2 * (precision * recall) / (precision + recall)
  print('Precision: ', precision_test, '\n', 'Recall: ', recall_test,'\n', 'F1-score:', f1_test)
    
except:
  print("cal recall_score error")
  pass

# COMMAND ----------

import numpy as np
from keras.utils import to_categorical
data = np.array([1, 5, 3, 8])
print(data)
def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

encoded_data = encode(data)
print(encoded_data)

def decode(datum):
    return np.argmax(datum)

for i in range(encoded_data.shape[0]):
    datum = encoded_data[i]
    print('index: %d' % i)
    print('encoded datum: %s' % datum)
    decoded_datum = decode(encoded_data[i])
    print('decoded datum: %s' % decoded_datum)
    print()

# COMMAND ----------

print(label_array)

# COMMAND ----------

for i in range(y_pred_test.shape[0]):
    datum = y_pred_test[i]
    print('index: %d' % i)
    print('encoded datum: %s' % datum)
    decoded_datum = decode(y_pred_test[i])
    print('decoded datum: %s' % decoded_datum)
    print()

# COMMAND ----------

y_pred_test = model.predict(seq_array)

# COMMAND ----------

print(y_pred_test)

# COMMAND ----------

class_names = [0,1,2]
plot_confusion_matrix(y_true, y_pred, classes=class_names, title='Confusion matrix, without normalization')

# COMMAND ----------

|

# COMMAND ----------

accPercent, precision, recall, f1_test = evaluationTest(epoch, model,seq_array,label_array,"models2/3_R_2_12_BDLSTM_RV5.h5", "3_R_2_12_BDLSTM_RV5.h5")

# COMMAND ----------

accPercent, precision, recall, f1_test = evaluationValidationSet(epoch,model,test_df,modelPath,modelFullName,chartPath)
parameters = [epoch, setN, modelPath,'BDLSTM', modelFullName , accPercent, precision*100, recall*100, f1_test*100]

# COMMAND ----------

print(modelResultDicBDLSTM)

# COMMAND ----------

resultMetrixLSTM = pd.DataFrame.from_dict(modelResultDicBDLSTM, orient='index', columns=['Epoch','SetN', 'Path','Algo','SName','Accuracy', 'Precision', 'Recall', 'F1-score'])
resultMetrixLSTM = resultMetrixLSTM.sort_values([ 'Accuracy'])
display(resultMetrixLSTM)

# COMMAND ----------

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(resultMetrixLSTM['Acc']), numpy.std(resultMetrixLSTM['Acc'])))

# COMMAND ----------

resultMetrixLSTM.to_csv("result_Test_BDLSTM.csv")
fromPathCSV = "file:/databricks/driver/result_Test_BDLSTM.csv"

try:
  print("Copying file [",fromPathCSV,"] to Data Lake....")
  copyData(fromPathCSV, dataLake+"/",False)
except:
  print("Error while trying to transfer file "+"file:/databricks/driver/"+fromPathCSV," to ",dataLake+"/")
  pass

# COMMAND ----------

# try:
#   evaluationTest(processCode, model,seq_array,history, label_array,currentModelPath)
# except:
#   print("Error Saving Model")
#   pass

# COMMAND ----------

# print(train_df['LABEL2'].value_counts())
# print('\nClass 0 samples =  {0:.0%}'.format(train_df['LABEL2'].value_counts()[0]/train_df['LABEL2'].count()))
# print('\nClass 1 samples =  {0:.0%}'.format(train_df['LABEL2'].value_counts()[1]/train_df['LABEL2'].count()))
# print('\nClass 2 samples =  {0:.0%}'.format(train_df['LABEL2'].value_counts()[2]/train_df['LABEL2'].count()))

# print(val_df['LABEL2'].value_counts())
# print('\nClass 0 samples =  {0:.0%}'.format(val_df['LABEL2'].value_counts()[0]/val_df['LABEL2'].count()))
# print('\nClass 1 samples =  {0:.0%}'.format(val_df['LABEL2'].value_counts()[1]/val_df['LABEL2'].count()))
# print('\nClass 2 samples =  {0:.0%}'.format(val_df['LABEL2'].value_counts()[2]/val_df['LABEL2'].count()))

# COMMAND ----------

# # print stat for binary classification label
# print(train_df['LABEL2'].value_counts())
# print('\nNegaitve samples =  {0:.0%}'.format(train_df['LABEL2'].value_counts()[0]/train_df['LABEL2'].count()))
# print('\nPosiitve samples =  {0:.0%}'.format(train_df['LABEL2'].value_counts()[1]/train_df['LABEL2'].count()))

# COMMAND ----------

## Stop Tensorboard
dbutils.tensorboard.stop()

# COMMAND ----------

### End Experiment ######