# Databricks notebook source
# MAGIC %md ### 
# MAGIC # Estimation of RUL using CNN
# MAGIC Predict if an asset will fail within certain time frame (e.g. cycles)

# COMMAND ----------

# MAGIC %md ## Deep Learining - CNN

# COMMAND ----------

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

import tensorflow as tf
from sklearn.preprocessing import scale

# define path to save model
model_path = 'binary_model.h5'

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

from keras import backend as K
import tensorflow as tf
with K.tf.device('/gpu:1'):
    config = tf.ConfigProto(intra_op_parallelism_threads=4,\
           inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU' :1, 'GPU' : 1})
    session = tf.Session(config=config)
    K.set_session(session)

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

def windows(nrows, size):
    start,step = 0, 2
    while start < nrows:
        yield start, start + size
        start += step

def segment_signal(features,labels,window_size = 15):
    segments = np.empty((0,window_size))
    segment_labels = np.empty((0))
    nrows = len(features)
    for (start, end) in windows(nrows,window_size):
        if(len(train_df.iloc[start:end]) == window_size):
            segment = features[start:end].T  #Transpose to get segment of size 24 x 15 
            label = labels[(end-1)]
            segments = np.vstack([segments,segment]) 
            segment_labels = np.append(segment_labels,label)
    segments = segments.reshape(-1,24,window_size,1) # number of features  = 24 
    segment_labels = segment_labels.reshape(-1,1)
    return segments,segment_labels

# COMMAND ----------

#1 - import module
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
# import numpy
# import pandas

# COMMAND ----------

import logging
logger = logging.getLogger('RCFailurePredictionRNNLog')
hdlr = logging.FileHandler('RCFailurePredictionRNNLog.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.WARNING)


# COMMAND ----------

## File Part in HDP Server to be stored images of charts or graph or temp files
path = "dbfs:/RC/datasets/output/"
sparkAppName = "RCDataAnlaysisUCM"
train_data = "dbfs:/FileStore/tables/tmp_rc/UCM/set1/*.csv"
test_data = "dbfs:/FileStore/tables/tmp_rc/NMD/set2/*.csv"

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
 'BE_SD_TYPE']

train_df = df.drop(*columns_to_drop)
#display(train_df)

# COMMAND ----------

#display(train_df.groupBy("OBJECT_ID").count())

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
logger.info("Finish normalization train dataset!")

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
logger.info("Finish normalization test dataset!")

# COMMAND ----------

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

# COMMAND ----------

segments, labels = segment_signal(train_df[sequence_cols],train_df['RUL'])

# COMMAND ----------

train_test_split = np.random.rand(len(segments)) < 0.70
train_x = segments[train_test_split]
train_y = labels[train_test_split]
test_x = segments[~train_test_split]
test_y = labels[~train_test_split]

# COMMAND ----------

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def apply_conv(x,kernel_height,kernel_width,num_channels,depth):
    weights = weight_variable([kernel_height, kernel_width, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights,[1,1,1,1],padding="VALID"),biases))
    
def apply_max_pool(x,kernel_height,kernel_width,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1], strides=[1, 1, stride_size, 1], padding = "VALID")

# COMMAND ----------

num_labels = 1
batch_size = 10
num_hidden = 800
learning_rate = 0.0001
training_epochs = 30
input_height = 24
input_width = 15
num_channels = 1
total_batches = train_x.shape[0] // batch_size

# COMMAND ----------

X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

c = apply_conv(X, kernel_height = 24, kernel_width = 4, num_channels = 1, depth = 8) 
p = apply_max_pool(c,kernel_height = 1, kernel_width = 2, stride_size = 2) 
c = apply_conv(p, kernel_height = 1, kernel_width = 3, num_channels = 8, depth = 14) 
p = apply_max_pool(c,kernel_height = 1, kernel_width = 2, stride_size = 2) 

shape = p.get_shape().as_list()
flat = tf.reshape(p, [-1, shape[1] * shape[2] * shape[3]])

f_weights = weight_variable([shape[1] * shape[2] * shape[3], num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.add(tf.matmul(f, out_weights),out_biases)

# COMMAND ----------

cost_function = tf.reduce_mean(tf.square(y_- Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

# COMMAND ----------

with tf.Session() as session:
    tf.global_variables_initializer().run()
    print("Training set MSE")
    for epoch in range(training_epochs):
        for b in range(total_batches):    
            offset = (b * batch_size) % (train_x.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :, :, :]
            batch_y = train_y[offset:(offset + batch_size),:]
            _, c = session.run([optimizer, cost_function],feed_dict={X: batch_x, Y : batch_y})
            
        p_tr = session.run(y_, feed_dict={X:  train_x})
        tr_mse = tf.reduce_mean(tf.square(p_tr - train_y))
        print(session.run(tr_mse))

    p_ts = session.run(y_, feed_dict={X:  test_x})
    ts_mse = tf.reduce_mean(tf.square(p_ts - test_y))
    print("Test set MSE: %.4f" % session.run(ts_mse)) 

# COMMAND ----------

dbutils.fs.ls("/mnt/Exploratory/WCLD")

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/model_verify.png", "dbfs:/mnt/Exploratory/WCLD/")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al
# MAGIC 
# MAGIC %python
# MAGIC dbutils.fs.mv("file:/databricks/driver/binary_submit_test.csv", "dbfs:/mnt/Exploratory/WCLD/")
# MAGIC dbutils.fs.mv("file:/databricks/driver/binary_submit_train.csv", "dbfs:/mnt/Exploratory/WCLD/")
# MAGIC dbutils.fs.ls("/mnt/Exploratory/WCLD")

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/binary_submit_test.csv", "dbfs:/mnt/Exploratory/WCLD/")
dbutils.fs.mv("file:/databricks/driver/binary_submit_train.csv", "dbfs:/mnt/Exploratory/WCLD/")

# COMMAND ----------

dbutils.fs.ls("/mnt/Exploratory/WCLD")

# COMMAND ----------

