# Databricks notebook source
# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

# MAGIC %fs
# MAGIC cp -r file:/tmp/tfb_log_dir/ dbfs:/mnt/Exploratory/WCLD/12_BDLSTM/tensorboard_log/

# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------

try:
  dbutils.fs.cp("file:/databricks/driver/models_BILSTM_R1_h5.zip", "dbfs:/mnt/Exploratory/WCLD/")
except:
    print("File Not Found Error models_BILSTM_R1_h5!")
    pass

try:
  dbutils.fs.cp("file:/databricks/driver/models_BILSTM_R1_h5.zip", "dbfs:/mnt/Exploratory/WCLD/")
except:
    print("File Not Found Error models_BILSTM_R1_h5.zip!")
    pass

# COMMAND ----------

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# COMMAND ----------

# MAGIC %sh
# MAGIC tensorboard --logdir='/tmp/tfb_log_dir/RCPredictiveMA_09RC_DIFF_BATCH_LSTM_2019020405/'

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al /tmp/tfb_log_dir/RCPredictiveMA_09RC_DIFF_BATCH_LSTM_2019020405/

# COMMAND ----------

print(n)

# COMMAND ----------

import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# COMMAND ----------

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# COMMAND ----------

print(sess)

# COMMAND ----------

# display(sess)

# COMMAND ----------

import tensorflow as tf
print(tf.test.is_built_with_cuda())

# COMMAND ----------

# tf = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# tf.list_devices()

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi pmon

# COMMAND ----------

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# COMMAND ----------

# MAGIC %sh
# MAGIC sudo pip install gpustat

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi -q -g 0 -d UTILIZATION -l

# COMMAND ----------

# MAGIC %sh
# MAGIC gpustat  -cp

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

# MAGIC %sh
# MAGIC cd logs

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/model_loss.png", "dbfs:/mnt/Exploratory/WCLD/")

# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/PredictiveManteinanceRCTraining.csv", "dbfs:/mnt/Exploratory/WCLD/")
dbutils.fs.cp("file:/databricks/driver/PredictiveManteinanceRCValidation.csv", "dbfs:/mnt/Exploratory/WCLD/")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

cd logs

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /databricks/driver/

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir tmp_rc_data/

# COMMAND ----------

# MAGIC %sh
# MAGIC cd

# COMMAND ----------

# MAGIC %sh
# MAGIC cd .

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

import tensorflow as tf
with tf.device('/gpu:3'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /proc/cpuinfo - cpu info

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /proc/meminfo - memory information

# COMMAND ----------

# MAGIC %sh
# MAGIC uptime

# COMMAND ----------

import tensorflow as tf

print(tf.__version__)

# COMMAND ----------

import tensorflow as tf

print(tf.VERSION)

# COMMAND ----------

# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))

# COMMAND ----------

# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))

# COMMAND ----------

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)

# COMMAND ----------

# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))

# COMMAND ----------

import tensorflow as tf
print(tf.test.is_built_with_cuda())

# COMMAND ----------

# MAGIC %sh
# MAGIC nvidia-smi

# COMMAND ----------

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# COMMAND ----------

import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))

# COMMAND ----------

# MAGIC %sh
# MAGIC which tensorflow

# COMMAND ----------

# MAGIC %sh 
# MAGIC nano ~/.bashrc

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

# MAGIC %sh
# MAGIC  cuda

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install tensorflow-gpu

# COMMAND ----------

# %sh
# pip uninstall tensorflow

# COMMAND ----------

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

# COMMAND ----------

# MAGIC %sh
# MAGIC #sudo chmod -R 777 binary_model.h5
# MAGIC ls -al

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al RCPredictiveMA

# COMMAND ----------

