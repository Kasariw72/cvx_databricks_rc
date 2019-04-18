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
sparkAppName = "12_BDLSTM_RV5"
# define path to save model
model_path = sparkAppName+".h5"

dirPath = "file:/databricks/driver/"
dataLake = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
mntDatalake = "/mnt/Exploratory/WCLD/"+sparkAppName

tf_at_dl_dir = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName
dbFSDir = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV6"
tensorboardLogPath = "dbfs:/mnt/Exploratory/WCLD/"+sparkAppName+"/tensorboard_log"

timestamp = datetime.datetime.now().strftime("%Y%m%d%H")

expContext = sparkAppName+"_"+str(timestamp)
tfb_log_dir = tfb_log_dir+expContext
print(expContext)
# dbutils.fs.mkdirs(tfb_log_dir)
### Enable Tensorboard and save to the localtion
dbutils.tensorboard.start(tfb_log_dir)

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
'VIBRA_ENGINE', 'S1_RECY_VALVE', 'S1_SUCT_PRESS', 'S1_SUCT_TEMPE', 'S2_STAGE_DISC_PRESS', 'S2_SCRU_SCRUB_LEVEL', 'GAS_LIFT_HEAD_PRESS', 'IN_CONT_CONT_VALVE', 'IN_SEP_PRESS']

sensor_cols = ['LUBE_OIL_PRESS',  'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP', 'THROW_1_SUC_PRESS', 
               'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS', 'THROW_2_DISC_TEMP', 'THROW_2_SUC_PRESS', 'THROW_2_SUC_TEMP', 
               'THROW_3_DISC_PRESS', 'THROW_3_DISC_TEMP','THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP', 'THROW_4_DISC_PRESS', 
               'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP', 'CYL_2_TEMP', 'CYL_3_TEMP', 
               'CYL_4_TEMP', 'CYL_5_TEMP','CYL_6_TEMP', 'CYL_7_TEMP', 'CYL_8_TEMP',
               'CYL_9_TEMP', 'CYL_10_TEMP','CYL_11_TEMP', 'CYL_12_TEMP', 'LUBE_OIL_PRESS_ENGINE', 
               'MANI_PRESS', 'RIGHT_BANK_EXH_TEMP','SPEED','VIBRA_ENGINE', 'S1_RECY_VALVE', 
               'S1_SUCT_PRESS', 'S1_SUCT_TEMPE', 'S2_STAGE_DISC_PRESS', 'S2_SCRU_SCRUB_LEVEL', 'GAS_LIFT_HEAD_PRESS', 
               'IN_CONT_CONT_VALVE', 'IN_SEP_PRESS']

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
        
  #df.cache()
  #rint("Spark finishs caching dataframe!")
  
  selectColDF = df.selectExpr(selectedCols)
  selectColDF = convertStr2Int(selectColDF, intCols, sensor_cols)
  
  #compute dataframe using sql command via string
  selectColDF.createOrReplaceTempView("RC_DATA_TMP")
  #filter out the data during RC S/D
  selectColDF = spark.sql("SELECT * FROM RC_DATA_TMP WHERE RUL BETWEEN 1 AND 2880 ORDER BY YEAR, EVENT_ID, CYCLE")
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

# MAGIC %md ## Deep Learining - LSTM for Binary Classification Objective

# COMMAND ----------

# pick a large window size of 120 cycles
# Target 1 days using 120 time steps

sequence_length = 90
input_length = 90

sensor_av_cols = ["AVG_" + nm for nm in sensor_cols]
sensor_sd_cols = ["SD_" + nm for nm in sensor_cols]

sequence_cols1 = sensor_av_cols + sensor_sd_cols
sequence_cols = sequence_cols + sequence_cols1

#print(sequence_cols)
columns_to_drop = ['CODE','YEAR']

v_batch_size = 400
v_validation_split = 0.05
v_verbose = 2
#verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
  
v_LSTMUnitLayer1 = 90
v_LSTMUnitLayer2 = 30
v_Dropout = 0.2
v_maxEpoch = 3

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

# COMMAND ----------

def buildBinaryClassModel(algoName,old_weights,v_LSTMUnitLayer1,v_LSTMUnitLayer2,sequence_length,nb_features,nb_out,v_Dropout,v_maxEpoch,v_batch_size,seq_array, label_array,valSeq_array, valLabel_array,model):
  if len(old_weights)<=0:
    
    print("Created New Model : "+algoName," Algo Mode :",algoName)
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer1, return_sequences=True),input_shape=(sequence_length, nb_features),merge_mode='concat'))
    model.add(Dropout(v_Dropout))
    model.add(Bidirectional(LSTM(units=v_LSTMUnitLayer2, return_sequences=False))) 
    model.add(Dropout(v_Dropout))
    model.add(Dense(units=nb_out, activation='sigmoid'))
    
#     print("Created New Model : "+modelName)
#     model.add(LSTM(input_shape=(sequence_length, nb_features), units=v_LSTMUnitLayer1, return_sequences=True)) 
#     model.add(Dropout(v_Dropout))
#     model.add(LSTM(units=v_LSTMUnitLayer2, return_sequences=False)) 
#     model.add(Dropout(v_Dropout))
#     model.add(Dense(units=nb_out, activation='sigmoid'))
    
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
    #print("scores_test:",scores_test)
    y_true = valLabel_array
    y_pred = model.predict(valSeq_array)
    labels = np.argmax(y_pred, axis=-1)
    
    print("labels:",labels[0:20])
    lb = preprocessing.LabelBinarizer()
    lb2 = lb.fit_transform(labels)
    print("lb:",lb2)
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
  print("Start Train Model ",iterNum)
  
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
  
  #Hyperparameters
  print(modelCodeMap[modelCode])
    
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
                                                       valLabel_array,model)
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
                                                         valLabel_array,model)
  else:
    print("Error, please verify and input model name!")
    
  try:
    ##print(history.history.keys())
    old_weights = model.get_weights()
    
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
trainMap,totalEvents = loadDataSet("TRAIN")
print("Total Events in Traing Dataset : ",str(totalEvents))
# # trainMap = {1: [2091, 2101], 
# #  2: [712, 714], 
# #  3: [924, 950]}
# trainMap = {1: [2091, 2101, 2103, 2104, 2119, 538, 546, 574, 578, 580, 660, 672, 679, 693, 694, 704, 705, 707, 712, 714, 732, 780, 787, 791, 815, 819, 842, 849, 852, 857, 860, 863]}
print(trainMap)

#Validation Sets
#valMap , totalEvents = loadDataSet("VALI")
valMap = {1: [2091, 2101, 2103, 2104, 2119, 538, 546, 574, 578, 580, 584, 590, 596, 689, 589, 592, 599, 616, 623, 631], 2: [660, 672, 679, 693, 694, 704, 705, 707, 712, 714, 732, 735, 740, 750, 751, 779, 834, 737, 743, 746], 3: [780, 787, 791, 815, 819, 842, 849, 852, 857, 860, 863, 878, 907, 912, 917, 926, 924, 950, 953, 961], 4: [1012, 1016, 1027, 1030, 1032, 1038, 1036, 1041, 1047, 1050, 1055, 1059, 1062, 1065, 1069, 1074, 1086, 1088, 1097, 1098], 5: [1115, 1120, 1122, 1128, 1133, 1134, 1144, 1149, 1162, 1165, 1172, 1174, 1189, 1228, 1229, 1251, 1259, 1262, 1187, 1202], 6: [1215, 1222, 1226, 1260, 1261, 1272, 1277, 1280, 1292, 1293, 1299, 1309, 1310, 1316, 1317, 1321, 1322, 1325, 1327, 1337], 7: [1346, 1355, 1356, 1362, 1363, 1365, 1368, 1372, 1375, 1379, 1380, 1381, 1383, 1404, 1399, 1400, 1402, 1403, 1409, 1416], 8: [1438, 1449, 1451, 1453, 1460, 1465, 1467, 1483, 1489, 1490, 1491, 1494, 1496, 1498, 1506, 1508, 1509, 1512, 1514, 1527], 9: [1535, 1550, 1559, 1580, 1530, 1532, 1536, 1541, 1545, 1552, 1560, 1567, 1576, 1584, 1585, 1590, 1596, 1601, 1602, 1603], 10: [1634, 1651, 1659, 1688, 1714, 1681, 1682, 1689, 1692, 1698, 1703, 1706, 1712, 1727, 1730, 1732, 1741, 1760, 1771, 1772], 11: [1800, 1822, 1826, 1828, 1847, 1864, 1836, 1844, 1845, 1851, 1852, 1867, 1872, 1875, 1882, 1886, 1887, 1893, 1894, 1910], 12: [1914, 1920, 1926, 1928, 1930, 1938, 1944, 1945, 1963, 1964, 1966, 1971, 1986, 1970, 1992, 1997, 2002, 2010, 2013, 2015], 13: [2019, 2020, 2022, 2028, 2029, 2036, 2040, 2048, 2052, 2069, 2070, 2077, 2078, 2108, 2110, 2142, 2143, 2146, 2149, 2156], 14: [2168, 2169, 2176, 2181, 2184, 2185, 2188, 2189, 2191, 2198, 2201, 2202, 2216, 2221, 2227, 2234, 2236, 2237, 2249, 2253], 15: [2276, 2278, 3843, 3844, 3850, 3871, 3875, 3877, 2247, 2263, 2266, 2268, 2275, 2281, 2286, 2301, 2306, 2316, 2319, 2329], 16: [2336, 2341, 2344, 2346, 2354, 2359, 2360, 2362, 2371, 2388, 2419, 2383, 2387, 2390, 2408, 2409, 2411, 2418, 2439, 2441], 17: [2455, 2459, 2465, 2471, 2482, 2488, 2491, 2496, 2503, 2515, 2517, 2521, 2529, 2533, 2535, 2536, 2543, 2545, 2555, 2559], 18: [2628, 2546, 2566, 2567, 2572, 2578, 2588, 2594, 2596, 2598, 2602, 2615, 2620, 2624, 2627, 2629, 2676, 2688, 2689, 2691], 19: [2708, 2714, 2715, 2740, 2744, 2747, 2761, 2806, 2728, 2729, 2734, 2738, 2743, 2746, 2750, 2753, 2757, 2765, 2771, 2777], 20: [2784, 2789, 2791, 2801, 2804, 2805, 2811, 2817, 2819, 2829, 2830, 2831, 2832, 2839, 2842, 2847, 2852, 2853, 2855, 2857], 21: [2870, 2871, 2891, 2893, 2952, 2953, 2875, 2878, 2888, 2892, 2903, 2904, 2910, 2923, 2926, 2929, 2931, 2939, 2949, 2950], 22: [2962, 2967, 2972, 2978, 2981, 2983, 2987, 2989, 2992, 2993, 2997, 2998, 3002, 3004, 3007, 3008, 3013, 3041, 3048, 3055], 23: [3050, 3057, 3060, 3061, 3063, 3075, 3091, 3092, 3093, 3103, 3106, 3107, 3108, 3109, 3119, 3121, 3124, 3129, 3130, 3134], 24: [3138, 3140, 3145, 3149, 3156, 3161, 3164, 3177, 3181, 3224, 3169, 3171, 3176, 3196, 3201, 3206, 3207, 3222, 3230, 3237], 25: [3270, 3271, 3274, 3278, 3283, 3291, 3303, 3337, 3340, 3341, 3344, 3351, 3358, 3364, 3381, 3382, 3384, 3387, 3397, 3409], 26: [3437, 3438, 3445, 3446, 3453, 3486, 3491, 3497, 3463, 3467, 3473, 3485, 3495, 3498, 3501, 3504, 3505, 3528, 3533, 3536], 27: [3557, 3558, 3559, 3568, 3573, 3579, 3581, 3582, 3590, 3595, 3597, 3617, 3615, 3616, 3621, 3629, 3631, 3635, 3636, 3639], 28: [3645, 3653, 3658, 3665, 3668, 3672, 3681, 3701, 3706, 3707, 3709, 3713, 3723, 3760, 3720, 3741, 3743, 3755, 3766, 3767], 29: [3774, 3777, 3781, 3783, 3791, 3792, 3793, 3794, 3797, 3801, 3812, 3818, 3823, 3824, 3826, 3830, 3837, 3839, 3859, 3864], 30: [3873, 3881, 3885, 3886, 3887, 3893, 3899, 3909, 3922, 3924, 3925, 3932, 3936, 3938, 3942, 3944, 3946, 3964, 3971, 3978], 31: [3986, 3987, 3995, 4005, 4013, 4031, 4004, 4007, 4010, 4027, 4033, 4036, 4040, 4042, 4044, 4049, 4052, 4061, 4065, 4074], 32: [4098, 4103, 4108, 4112, 4117, 4119, 4135, 4140, 4145, 4146, 4164, 4165, 4167, 4174, 4152, 4163, 4168, 4178, 4182, 4187], 33: [4217, 4221, 4223, 4242, 4254, 4263, 4267, 4272, 4276, 4277, 4281, 4290, 4295, 4297, 4301, 4373, 4381, 4390, 4394, 4402]}
#print("Total Events in Validation Dataset : ",str(totalEvents))
# valMap = {1: [635, 655, 760, 776]}
valMap = {1: [635, 655], 2: [760, 776], 3: [962, 964], 4: [1106, 1114], 5: [1204, 1209], 6: [1338, 1340], 7: [1420, 1429], 8: [1528, 1529], 9: [1626, 1633], 10: [1787, 1795], 11: [1911, 1912], 12: [2017, 2018], 13: [2158, 2164], 14: [2256, 2272], 15: [2332, 2335], 16: [2450, 2451], 17: [2568, 2587], 18: [2698, 2701], 19: [2779, 2783], 20: [2859, 2860], 21: [2957, 2960], 22: [3064, 3067], 23: [3135, 3137], 24: [3251, 3265], 25: [3424, 3431], 26: [3544, 3553], 27: [3641, 3642], 28: [3769, 3772], 29: [3868, 3872], 30: [3981, 3985], 31: [4078, 4092], 32: [4200, 4201], 33: [4411, 4412]}

#print(valMap)

# #Test Sets:
#testMap,totalEvents = loadDataSet("TEST")
#print("Total Events in Test Dataset : ",str(totalEvents))
testMap = {1: [4415, 4420, 4421, 4422, 4428, 4433]}
print(testMap)

# COMMAND ----------

def printDFPortion(train_df, val_df):
  
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
  
#   try:
#     print("+++++Train Data Set LABEL1 ++++++++++++")
#     # print stat for binary classification label
#     print(train_df['LABEL1'].value_counts())
#     print('\nNegaitve samples [0] = {0:.0%}'.format(train_df['LABEL1'].value_counts()[0]/train_df['LABEL1'].count()))
#     print('\nPosiitve samples [1] = {0:.0%}'.format(train_df['LABEL1'].value_counts()[1]/train_df['LABEL1'].count()))
#   except:
#     pass
  
#   try:
#     print("+++++Validation Data Set LABEL1 ++++++++++++")
#     print(val_df['LABEL1'].value_counts())
#     print('\nNegaitve samples [0] = {0:.0%}'.format(val_df['LABEL1'].value_counts()[0]/val_df['LABEL1'].count()))
#     print('\nPosiitve samples [1] = {0:.0%}'.format(val_df['LABEL1'].value_counts()[1]/val_df['LABEL1'].count()))
#   except:
#     pass
  

# COMMAND ----------

## Aj. Suggested to use only one set of UCM event to test model
maxBatch = len(trainMap)
val_df = ""
isNotSatisfacEval = True
old_weights =""

#min_max_scaler = preprocessing.MinMaxScaler()
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
  curLoss =0.01
  curMAcc = 0.95
  curVAcc = 0.95
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

model = Sequential()
#lastModelPath = "models/223_R_25_12_BDLSTM_RV4.h5"
old_weights = ""

# if os.path.isfile(lastModelPath):
#   ##estimator = load_model(modelPath)
#   model = load_model(lastModelPath)
#   old_weights = model.get_weights()
#   print("Model Loaded!",lastModelPath)
# else:
#   print("Model File Not Found!",lastModelPath)

# COMMAND ----------

for nEpoch in range(nEpochs):
  
  print("Starting Loop : ",str(cLoop))
  
  countTrainSet = 1
  trainDataSetKeys = trainMap.keys()
  valDataSetKeys = valMap.keys()
  
  for trainKey in trainDataSetKeys:
    if (trainKey>=0 and nEpoch>0):
      cLoop = cLoop+1
      if isNotSatisfacEval is True:
        print("Train model using dataset {",str(trainKey),"}")
        train_df = getDataFromCSV(sqlContext, dbFSDir,trainMap[trainKey], selectedCols)
        train_df = train_df.append(train_df)
        train_df = train_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
        train_df = add_features(train_df, rolling_win_size , sensor_cols)
        train_df = train_df.drop(columns=columns_to_drop)
        
        if len(test_df)>0:
          print("Test data is in memory!")
        else:
          print("Loading test data.")
          testDataSetKeys = testMap.keys()
          for testKey in testDataSetKeys:
            test_df = getDataFromCSV(sqlContext, dbFSDir,testMap[testKey], selectedCols)
            test_df = test_df.append(test_df)
            break

          test_df = test_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
          test_df = add_features(test_df, rolling_win_size , sensor_cols)
          test_df = test_df.drop(columns=columns_to_drop)

        if trainKey in valDataSetKeys:
          print("Loading Validate dataset[",trainKey,"]",valMap[trainKey])
          val_df = getDataFromCSV(sqlContext, dbFSDir,valMap[trainKey], selectedCols)
          val_df = val_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
          val_df = add_features(val_df, rolling_win_size , sensor_cols)
          val_df = val_df.drop(columns=columns_to_drop)
        else:
          print("Key not found in Validation Map, please verify the index file.")
        
        printDFPortion(train_df, val_df)
        processCode = str(cLoop)+"_R_"+str(countTrainSet)
        history, old_weights, model = trainModel("BDLSTM",old_weights,train_df,val_df, test_df ,processCode,tensor_board,model)
        
        if len(old_weights)==0:
          print("Error Empty Weights!!")
        else:
          print("Has weights!!")
        
        try:
          lossOptimal, score, resutl, curVLoss = isOptimal(history,countTrainSet,score,curVLoss,nEpoch)
        except:
          print("Error in lossOptimal..")
          pass

        resultMetric[cLoop] = [processCode] + resutl
        print(resultMetric)

        if lossOptimal is False:
          countTrainSet=countTrainSet+1
        else:
          break  

      else:
        print("Train and evaluation is satisfactory!")
        break
        
##run_train_all()

# COMMAND ----------

resultMetrixLSTM = pd.DataFrame.from_dict(resultMetric, orient='index', columns=['m_code', 'Loss', 'Acc', 'Val_Loss', 'Val_Acc'])
resultMetrixLSTM = resultMetrixLSTM.sort_values([ 'm_code'])
print("%.2f%% (+/- %.2f%%)" % (np.mean(resultMetrixLSTM['Acc']), np.std(resultMetrixLSTM['Acc'])))

resultMetrixLSTM.to_csv("result_BDLSTM.csv")
fromPathCSV = "file:/databricks/driver/result_BDLSTM.csv"

try:
  print("Copying file [",fromPathCSV,"] to Data Lake....")
  copyData(fromPathCSV, dataLake+"/",False)
except:
  print("Error while trying to transfer file "+"file:/databricks/driver/"+fromPathCSV," to ",dataLake+"/")
  pass

try:
  dbutils.fs.cp("file:/tmp/tfb_log_dir/", "dbfs:/mnt/Exploratory/WCLD/12_BDLSTM_RV5/tensorboard_log/",recurse=True)
except:
  print("Not found tensorboard file or copying error!")
  pass

# COMMAND ----------

try:
  print("Copying file tensorboard to Data Lake....")
  copyData("file:/"+tfb_log_dir, tensorboardLogPath ,True)
except:
  print("Error while trying to transfer tensorboard file.")
  pass

# COMMAND ----------

def loadTest():
  train_df = getDataFromCSV(sqlContext, dbFSDir,trainMap[1], selectedCols)
  train_df = train_df.append(train_df)
  train_df = train_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
  train_df = add_features(train_df, rolling_win_size , sensor_cols)
  train_df = train_df.drop(columns=columns_to_drop)

  print("Loaded train dataset")
  val_df = getDataFromCSV(sqlContext, dbFSDir,valMap[1], selectedCols)
  val_df = val_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
  val_df = add_features(val_df, rolling_win_size , sensor_cols)
  val_df = val_df.drop(columns=columns_to_drop)

  print("Loaded validate dataset")
  test_df = getDataFromCSV(sqlContext, dbFSDir,testMap[1], selectedCols)
  test_df = test_df.append(test_df)
  test_df = test_df.sort_values(['CODE','YEAR','EVENT_ID','CYCLE'])
  test_df = add_features(test_df, rolling_win_size , sensor_cols)
  test_df = test_df.drop(columns=columns_to_drop)
  print("Loaded test dataset")
  return train_df,val_df,test_df

# COMMAND ----------

def testGen():
  targetColName = "LABEL2"
  ##print(sequence_cols)
  seq_array, label_array, nb_features, nb_out = gen_data_train_val(targetColName, train_df,sequence_length, sequence_cols)
  print("Finish Gen Train Data Sequence")

  valSeq_array, valLabel_array, valNb_features, valNb_out = gen_data_train_val(targetColName, val_df,sequence_length, sequence_cols)
  print("Finish Gen Validate Data Sequence")

  testSeq_array, testLabel_array, testNb_features, testNb_out = gen_data_train_val("LABEL2", test_df,sequence_length, sequence_cols)

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
  test_df2 = getDataFromCSV(sqlContext, dbFSDir,testMap[1], selectedCols)
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

def evaluationTest2():
  try:
    scores_test = model.evaluate(valSeq_array, valLabel_array, verbose=1)
    print("scores_test:",scores_test)
    y_true = valLabel_array
    y_pred = model.predict_classes(valSeq_array)
    categorical_acc = keras.metrics.categorical_accuracy(y_true, y_pred)

    labels = np.argmax(probas, axis=-1)
    lb = preprocessing.LabelBinarizer()
    lb.fit_transform(labels)

    print("y_true[0:20]", y_true[0:20])
    print("y_pred[0:20]", y_pred[0:20])

    print('Evaluation Accurracy: {}'.format(round_down_float(scores_test[1]*100,2)))
    print('Categorical Accurracy:',categorical_acc)

    try:
      cm = confusion_matrix(y_target=y_true, 
                        y_predicted=y_pred, 
                        binary=False)

      #print('Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')
      #cm = confusion_matrix(y_true, y_pred)
      print(cm)
    except:
      print("Error CM1")
      pass

    try:
      y_true_non_category = [ np.argmax(t) for t in y_true]
      y_predict_non_category = [ np.argmax(t) for t in y_pred ]

      print("y_true_non_category[0:20]", y_true_non_category[0:20])
      print("y_predict_non_category[0:20]", y_predict_non_category[0:20])

      conf_mat = confusion_matrix(y_target=y_true_non_category, 
                        y_predicted=y_predict_non_category, 
                        binary=False)

      #conf_mat = confusion_matrix(y_true_non_category, y_predict_non_category)
      print(conf_mat)
    except:
      print("Error CM2")
      pass

    try:
      fbeta = fbeta_score(y_true_non_category, y_predict_non_category, beta=1)
      fm = fmeasure(y_true_non_category, y_predict_non_category)
      print('fbeta :',fbeta)
      print('fmeasure :',fm)
    except:
      print("Error in calculating F Score.")
      pass
  except:
    pass

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

try:
  plot_history(history,"History categorical_accuracy ")
except:
  pass

# COMMAND ----------

# fbeta = fbeta_score(y_true, y_pred, beta=1)
# fm = fmeasure(y_true, y_pred)

# COMMAND ----------

# %fs
# cp dbfs:/mnt/Exploratory/WCLD/12_BDLSTM_RV4/model/157_R_25_12_BDLSTM_RV4.h5 file:/databricks/driver

# COMMAND ----------

## Stop Tensorboard
dbutils.tensorboard.stop()

# COMMAND ----------

### End Experiment ######