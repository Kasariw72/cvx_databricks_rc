# Databricks notebook source
# MAGIC %md ### 
# MAGIC # Regression
# MAGIC How many more cycles an in-service engine will last before it fails?

# COMMAND ----------

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
expContext = "02_RC_LSTM_REG_"+str(timestamp)+"_"

print(expContext)
log_dir = "tmp/tensorflow_"+expContext

# COMMAND ----------

### Enable Tensorboard and save to the localtion
dbutils.tensorboard.start(log_dir)

# COMMAND ----------

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import preprocessing

# Setting seed for reproducibility
np.random.seed(1234)  
PYTHONHASHSEED = 0

# define path to save model
model_path = 'regression_model.h5'

# COMMAND ----------

from keras import backend as K
import tensorflow as tf

with K.tf.device('/gpu:1'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU' :1, 'GPU':1})
    session = tf.Session(config=config)
    K.set_session(session)

# COMMAND ----------

#%load_ext autoreload
#%autoreload 2
#%matplotlib inline
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

#jupyter wdgets
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

# import logging
# logger = logging.getLogger('RCFailurePredictionRNNLog')
# hdlr = logging.FileHandler('RCFailurePredictionRNNLog.log')
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

# hdlr.setFormatter(formatter)
# logger.addHandler(hdlr) 
# logger.setLevel(logging.WARNING)


# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "dbfs:/RC/datasets/output/"
sparkAppName = "RCDataAnlaysisUCM"
train_data = "dbfs:/FileStore/tables/tmp_rc/UCM/set1/*.csv"
test_data = "dbfs:/FileStore/tables/tmp_rc/UCM/set2/*.csv"

dirPath = "file:/databricks/driver/"
dataLake = "dbfs:/mnt/Exploratory/WCLD/"

# COMMAND ----------

# MAGIC %md ## Data Ingestion
# MAGIC Training Data

# COMMAND ----------

# read training data - It is the RC engine run-to-failure data.
#trainDataSetPath  "dbfs:/RC/datasets/train/*.csv"
df = spark.read.format("csv").option("inferSchema", "true").option("header","true").load(train_data)

# COMMAND ----------


columns_to_drop = ['_c0',
 'CODE',
 'DAYTIME',
 'YEAR',
 'MONTH',
 'DAY',
 'HOUR',
 'MM',
 'SD_TYPE',
 'IN_RUL',
 'DIGIT_SD_TYPE',
 'LABEL1',
 'LABEL2',
 'BE_SD_TYPE']

train_df = df.drop(*columns_to_drop)
#display(train_df)

# COMMAND ----------

# import seaborn as sns
# sns.set(style="white")

# #df = sns.load_dataset("iris")
# g = sns.PairGrid(train_df, diag_sharey=False)
# g.map_lower(sns.kdeplot)
# g.map_diag(sns.kdeplot, lw=3)
# g.map_upper(sns.regplot)

# display(g.fig)

# COMMAND ----------

train_df = train_df.toPandas()

# COMMAND ----------

train_df = train_df.sort_values(['OBJECT_ID', 'EVENT_ID', 'CYCLE'])
#display(train_df.head())

# COMMAND ----------

# MAGIC %md ## Data Ingestion
# MAGIC Test Data

# COMMAND ----------

test_df = spark.read.format("csv").option("inferSchema", "true").option("header","true").load(test_data)

# COMMAND ----------

columns_to_drop = ['_c0',
 'CODE',
 'DAYTIME',
 'YEAR',
 'MONTH',
 'DAY',
 'HOUR',
 'MM',
 'SD_TYPE',
 'LABEL1',
 'LABEL2',
 'IN_RUL',
 'DIGIT_SD_TYPE',
 'BE_SD_TYPE']
test_df = test_df.drop(*columns_to_drop)

# 'THROW_2_SUC_PRESS',
#display(test_pd.head())

# COMMAND ----------

test_df = test_df.toPandas()
#display(test_df.head())

# COMMAND ----------

#display(test_df.groupBy("OBJECT_ID").count())
test_df = test_df.sort_values(['OBJECT_ID', 'EVENT_ID', 'CYCLE'])

# COMMAND ----------

# MAGIC  %md ## MinMax normalization

# COMMAND ----------

# MinMax normalization (from 0 to 1)
train_df['cycle_norm'] = train_df['CYCLE']
cols_normalize = train_df.columns.difference(['EVENT_ID','OBJECT_ID','CYCLE','RUL','LABEL1','LABEL2'])

min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)

print("Finish normalization train dataset!")
##logger.info("Finish normalization train dataset!")

# COMMAND ----------

######
# TEST
######
# MinMax normalization (from 0 to 1)
test_df['cycle_norm'] = test_df['CYCLE']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)

print("Finish normalization test dataset!")
#logger.info("Finish normalization test dataset!")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)

# COMMAND ----------

##################################
# Data Preprocessing
##################################

#######
# TRAIN
#######
# Data Labeling - generate column RUL(Remaining Usefull Life or Time to Failure)
rul = pd.DataFrame(train_df.groupby('EVENT_ID')['CYCLE'].max()).reset_index()

rul.columns = ['EVENT_ID', 'max']
train_df = train_df.merge(rul, on=['EVENT_ID'], how='left')
train_df['RUL'] = train_df['max'] - train_df['CYCLE']
train_df.drop('max', axis=1, inplace=True)

# COMMAND ----------

# display(train_df.head())

# COMMAND ----------

# generate label columns for training data
# we will only make use of "label1" for binary classification, 
# while trying to answer the question: is a specific engine going to fail within w1 cycles?
w1 = 600
w0 = 300
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

# COMMAND ----------

# columns_to_drop = ['LABEL1','LABEL2']

# train_df = train_df.drop(*columns_to_drop)

# COMMAND ----------

# display(train_df.head())

# COMMAND ----------

datafile = expContext+"PredictiveManteinanceRCTraining.csv"

train_df.to_csv(datafile, encoding='utf-8',index = None)
dbutils.fs.mv(dirPath+datafile, dataLake)

# COMMAND ----------

##train_df.to_csv('PredictiveManteinanceRCTraining.csv', encoding='utf-8',index = None)
######
# TEST
######
# MinMax normalization (from 0 to 1)
test_df['cycle_norm'] = test_df['CYCLE']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)
print(test_df.head())

# # We use the ground truth dataset to generate labels for the test data.
# # generate column max for test data
rul = pd.DataFrame(test_df.groupby('EVENT_ID')['CYCLE'].max()).reset_index()
rul.columns = ['EVENT_ID', 'max']
# truth_df.columns = ['more']
# truth_df['EVENT_ID'] = truth_df.index + 1
# truth_df['max'] = rul['max'] + truth_df['more']
# truth_df.drop('more', axis=1, inplace=True)

# generate RUL for test data
#test_df = test_df.merge(truth_df, on=['EVENT_ID'], how='left')
# test_df['RUL'] = test_df['max'] - test_df['CYCLE']
# test_df.drop('max', axis=1, inplace=True)

# generate label columns w0 and w1 for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

datafile = expContext+"PredictiveManteinanceRCValidation.csv"

train_df.to_csv(datafile, encoding='utf-8',index = None)
dbutils.fs.mv(dirPath+datafile, dataLake)


# COMMAND ----------


# pick a large window size of 50 cycles
sequence_length = 50

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
        
# pick the feature columns 
##sensor_cols = ['s' + str(i) for i in range(1,22)]
##sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
#sequence_cols.extend(sensor_cols)

sequence_cols = ['RESI_LL_LUBE_OIL_PRESS', 'RESI_HH_LUBE_OIL_TEMP', 'RESI_LL_SPEED',
'RESI_HH_SPEED', 'RESI_LL_VIBRATION', 'RESI_HH_VIBRATION',
'RESI_HH_THROW1_DIS_TEMP', 'RESI_HH_THROW1_SUC_TEMP', 'LUBE_OIL_PRESS',
'LUBE_OIL_TEMP', 'THROW_1_DISC_PRESS', 'THROW_1_DISC_TEMP',
'THROW_1_SUC_PRESS', 'THROW_1_SUC_TEMP', 'THROW_2_DISC_PRESS',
'THROW_2_DISC_TEMP', 'THROW_2_SUC_TEMP', 'THROW_3_DISC_PRESS',
'THROW_3_DISC_TEMP', 'THROW_3_SUC_PRESS', 'THROW_3_SUC_TEMP',
'THROW_4_DISC_PRESS', 'THROW_4_DISC_TEMP', 'VIBRATION', 'CYL_1_TEMP',
'CYL_2_TEMP', 'CYL_3_TEMP', 'CYL_4_TEMP', 'CYL_5_TEMP', 'CYL_6_TEMP',
'CYL_7_TEMP', 'CYL_8_TEMP', 'CYL_9_TEMP', 'CYL_10_TEMP', 'CYL_11_TEMP',
'FUEL_GAS_PRESS', 'LUBE_OIL_PRESS_ENGINE', 'SPEED', 'VIBRA_ENGINE','cycle_norm']

# TODO for debug 
# val is a list of 192 - 50 = 142 bi-dimensional array (50 rows x 25 columns)
val=list(gen_sequence(train_df[train_df['EVENT_ID']==1], sequence_length, sequence_cols))
print(len(val))

# generator for the sequences
# transform each id of the train dataset in a sequence
seq_gen = (list(gen_sequence(train_df[train_df['EVENT_ID']==id], sequence_length, sequence_cols)) 
           for id in train_df['EVENT_ID'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape)

# function to generate labels
def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
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

# generate labels
label_gen = [gen_labels(train_df[train_df['EVENT_ID']==id], sequence_length, ['RUL']) 
             for id in train_df['EVENT_ID'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)
label_array.shape

# COMMAND ----------

# MAGIC %md ## Deep Learining - LSTM for Regression Objective

# COMMAND ----------

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Next, we build a deep network. 
# The first layer is an LSTM layer with 100 units followed by another LSTM layer with 50 units. 
# Dropout is also applied after each LSTM layer to control overfitting. 
# Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.

nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()
model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=80,
         return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
          units=40,
          return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out))
model.add(Activation("linear"))
model.compile(loss='mean_squared_error', optimizer='rmsprop',metrics=['mae',r2_keras])

print(model.summary())

# COMMAND ----------

# fit the network
history = model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())

# COMMAND ----------

# MAGIC %md ## Model Evaluation on Validation set

# COMMAND ----------

# summarize history for R^2
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['r2_keras'])
plt.plot(history.history['val_r2_keras'])
plt.title('model r^2')
plt.ylabel('R^2')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
display(fig_acc)

datafile = expContext+"LSTM_REG_model_r2.png"
fig_acc.savefig(datafile)
dbutils.fs.mv(dirPath+datafile, dataLake)


# COMMAND ----------

##summarize history for MAE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
display(fig_acc)

datafile = expContext+"LSTM_REG_model_mae.png"
fig_acc.savefig(datafile)
dbutils.fs.mv(dirPath+datafile, dataLake)

# COMMAND ----------

display(fig_acc)

# COMMAND ----------

# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

##plt.show()
fig_acc.savefig("LSTM_REG_model_regression_loss.png")

datafile = expContext+"LSTM_REG_model_regression_loss.png"
dbutils.fs.mv(dirPath+datafile, dataLake)

# training metrics
scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
print('\nMAE: {}'.format(scores[1]))
print('\nR^2: {}'.format(scores[2]))

y_pred = model.predict(seq_array,verbose=1, batch_size=200)
y_true = label_array

test_set = pd.DataFrame(y_pred)

datafile = expContext+"LSTM_REG_submit_train.csv"
test_set.to_csv(datafile, index = None)
dbutils.fs.mv(dirPath+datafile, dataLake)


# COMMAND ----------

display(fig_acc)

# COMMAND ----------

## Stop Tensorboard
dbutils.tensorboard.stop()

# COMMAND ----------

# dbutils.fs.ls('File:/tmp/tensorflow_log_dir')

# COMMAND ----------

import shutil
shutil.mv(log_dir, "/dbfs/tensorflow/logs")
# dbutils.fs.mv("File:/tmp/tensorflow_log_dir/events.out.tfevents.1548840731.0122-055838-bets960-10-139-64-8", "dbfs:/mnt/Exploratory/WCLD/tensorflow/logs/events.out.tfevents.1548840731.0122-055838-bets960-10-139-64-8")

# COMMAND ----------

# dbutils.fs.ls("/mnt/Exploratory/WCLD")

# COMMAND ----------

# dbutils.fs.mv("file:/databricks/driver/LSTM_REG_submit_train.csv", "dbfs:/mnt/Exploratory/WCLD/")
# dbutils.fs.mv("file:/databricks/driver/LSTM_REG_model_regression_loss.png", "dbfs:/mnt/Exploratory/WCLD/")