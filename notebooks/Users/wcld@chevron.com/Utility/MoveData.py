# Databricks notebook source
# MAGIC %sh
# MAGIC ls MAWD

# COMMAND ----------

# MAGIC %sh
# MAGIC cp datafile/*MAWD* MAWD

# COMMAND ----------

# MAGIC %fs
# MAGIC cp -r file:/databricks/driver/MAWD dbfs:/RC/RC_EVENTS/ALL_FILES_RV6

# COMMAND ----------

# MAGIC %fs
# MAGIC cp -r file:/databricks/driver/datafile dbfs:/RC/RC_EVENTS/ALL_FILES_RV6

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
  modelList.insert(2,line[1])
  modelMap[modelWords[0]] = modelList
  
print(modelMap)

# COMMAND ----------

modelMapPD = pd.DataFrame.from_dict(modelMap, orient='index', columns=['Epoch', 'SetN', 'MName'])
modelMapPD = modelMapPD.sort_values([ 'Epoch'])
display(modelMapPD)

# COMMAND ----------

def copyData(fromPath, toPath,recurseFlag=True):
  dbutils.fs.cp(fromPath, toPath,recurse=recurseFlag)
  print("Completed copying ["+fromPath+"] to "+toPath)

# COMMAND ----------

dbutils.fs.cp("file:/tmp/tfb_log_dir/", "dbfs:/mnt/Exploratory/WCLD/12_BDLSTM/tensorboard_log/",recurse=True)
dbutils.fs.cp("file:/"+tfb_log_dir , dataLake+"/tensorboard_log/",recurse=True)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls

# COMMAND ----------

a = "dbfs:/mnt/Exploratory/WCLD/dataset/EVENT_ID_DATA_SETS_FULL_V9.csv"
b = "dbfs:/RC/MSATER_DATA_FILES/"
copyData(a,b,False)

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/RC/MSATER_DATA_FILES/

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir datafile

# COMMAND ----------

# MAGIC %sh
# MAGIC zip LSTM_R6_RCPredictiveMA.zip *.h5

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

# MAGIC %sh 
# MAGIC #mkdir img_result_20190206
# MAGIC #cp *.png img_result_20190206
# MAGIC cd img_result_20190206

# COMMAND ----------

try:
  dbutils.fs.cp("file:/databricks/driver/LSTM_R6_RCPredictiveMA.zip", "dbfs:/mnt/Exploratory/WCLD/")
  dbutils.fs.cp("file:/databricks/driver/LSTM_R6_RCPredictiveMA.zip", "dbfs:/RC/")
except:
    print("File Not Found Error.")
    pass

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir model_9m_8

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/BetaProject/models_9_6_h5.zip file:/databricks/driver

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip models_9_6_h5.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC rm *.h5

# COMMAND ----------

# MAGIC %sh
# MAGIC ls *.h5

# COMMAND ----------

import shutil
#shutil.move(expContext, "/dbfs/FileStore/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM/"+expContext)
 
try:
  #shutil.move("traing_results.zip", "/dbfs/FileStore/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM")
  shutil.move("img_result_20190206", "/dbfs/FileStore/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM/9_2_RC_DIFF_BATCH_IMG")
except:
  print(" Error Move Folder!")
  pass

# COMMAND ----------

resultPath = "/FileStore/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM/img_result_20190206"

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al /tmp/tfb_log_dir/09_6_RC_DIFF_BATCH_LSTM_2019021004

# COMMAND ----------

# MAGIC %sh
# MAGIC zip tensorboard_data.zip /tmp/tfb_log_dir/09_6_RC_DIFF_BATCH_LSTM_2019021004

# COMMAND ----------

# MAGIC %sh
# MAGIC zip tensorboard_data.zip /tmp/tfb_log_dir/09_6_RC_DIFF_BATCH_LSTM_2019021004/*.*

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/tensorboard_data.zip dbfs:/RC/

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/11_BDLSTM/BILSTM_RCPredictiveMA_RV02_20190301_R2.zip file:/databricks/driver/

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /databricks/driver/BILSTM_RCPredictiveMA_RV02_20190301_R2.zip

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/LSTM_R6_RCPredictiveMA.zip file:/databricks/driver/

# COMMAND ----------

file = "LSTM_R6_RCPredictiveMA.zip"
datalake = "dbfs:/mnt/Exploratory/WCLD/"
localpath = "file:/databricks/driver/"

moveData(datalake+file, localpath,file)

# COMMAND ----------

def moveData(fromPath, toPath,fileName):
  localPath = "/databricks/driver/"
  dbutils.fs.cp(fromPath, toPath)
  !unzip toPath+fileName
  !ls -al *.h5

# COMMAND ----------

# MAGIC %sh
# MAGIC ls *.h5

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip /databricks/driver/LSTM_R6_RCPredictiveMA.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC ls *.h5

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip LSTM_R5_RCPredictiveMA.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC ls *LSTM_R3_RCPredictiveMA.h5

# COMMAND ----------

# MAGIC %sh
# MAGIC ls *.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC zip binary_submit_test.zip *.csv

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/binary_submit_test.zip dbfs:/mnt/Exploratory/WCLD/dataset

# COMMAND ----------

# MAGIC %sh
# MAGIC zip image_LSTM_RV6.zip *.png

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/image_LSTM_RV6.zip dbfs:/mnt/Exploratory/WCLD/

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/mnt/Exploratory/WCLD/dataset

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/dataset/RC_T_MAIN_DATA_06.zip file:/databricks/driver/

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip RC_T_MAIN_DATA_06.zip

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/dataset/RC_T_MAIN_DATA_09.zip file:/databricks/driver/

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip RC_T_MAIN_DATA_09.zip

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/dataset/RC_T_MAIN_DATA_10.zip file:/databricks/driver/

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/dataset/RC_T_MAIN_DATA05_RV1.zip file:/databricks/driver/

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip RC_T_MAIN_DATA_08.zip
# MAGIC unzip RC_T_MAIN_DATA_10.zip

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/RC_T_MAIN_DATA_08.csv dbfs:/RC/MSATER_DATA_FILES/

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/RC_T_MAIN_DATA_06.csv dbfs:/RC/MSATER_DATA_FILES/

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA04RV2.csv dbfs:/RC/MSATER_DATA_FILES/TMP_20190317_RC_T_MAIN_DATA04RV2.csv

# COMMAND ----------

# MAGIC %fs
# MAGIC rm dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA04RV2.csv

# COMMAND ----------

# MAGIC %sh 
# MAGIC unzip RC_T_MAIN_DATA05_RV1.zip

# COMMAND ----------

# MAGIC %fs
# MAGIC ls dbfs:/RC/MSATER_DATA_FILES/

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA05_RV3.csv dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA04.csv

# COMMAND ----------

# MAGIC %fs
# MAGIC rm dbfs:/RC/MSATER_DATA_FILES/RC_T_MAIN_DATA05_RV3.csv 

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/RC_T_MAIN_DATA05_RV1.csv dbfs:/RC/MSATER_DATA_FILES/

# COMMAND ----------

# MAGIC %fs
# MAGIC rm dbfs:/RC/MSATER_DATA_FILES/RC_T_SAMPLE_DATA_08RV2.csv

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/RC_T_SAMPLE_DATA_08RV2.csv dbfs:/RC/MSATER_DATA_FILES/

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al datafile

# COMMAND ----------

# MAGIC %sh
# MAGIC zip rc_event_data03.zip datafile/*.csv

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/dataset/EVENT_ID_DATA_SETS_01.csv dbfs:/RC/MSATER_DATA_FILES/

# COMMAND ----------

def moveData(fromPath, toPath):
  dbutils.fs.cp(fromPath, toPath,recurse=True)
  print("Complete copying files to "+toPath)

# COMMAND ----------

toPath = "dbfs:/RC/RC_EVENTS/ALL_FILES_RV2"
fromPath = "file:/databricks/driver/datafile"

moveData(fromPath,toPath)

# COMMAND ----------

pdDF = dbutils.fs.ls(fromPath)

# COMMAND ----------

sqlNotInId = [4799, 4623, 2052, 2111, 3387, 3222, 2847, 4478, 4741, 2503, 3505, 2387, 4078, 3145, 3774, 2496, 2247, 5806, 3830, 3707, 2017, 3048, 2715, 4883, 4854, 2029, 3641, 2923, 3635, 2801, 3061, 2989, 6709, 4412, 2040, 4042, 3653, 6257, 4895, 4061, 4833, 4932, 3573, 4082, 2189, 4500, 3817, 2535, 2286, 2269, 4890, 2432, 2830, 6027, 2842, 2176, 4912, 4953, 3781, 6512, 4449, 2383, 1368, 1506, 3582, 4494, 3568, 5917, 3887, 4445, 779, 2952, 4662, 2341, 1416, 4468, 1209, 3864, 1663, 1064, 6668, 2831, 1038, 3376, 955, 1409, 2158, 3251, 4103, 4201, 1383, 4373, 2701, 3115, 1260, 3381, 4489, 6858, 1187, 2376, 3364, 1453, 4653, 1948, 2284, 2522, 5043, 1644, 4646, 4516, 3105, 6103, 3671, 2914, 1384, 4492, 731, 2901, 2088, 827, 1960, 4822, 2458, 1053, 4382, 2312, 2293, 2034, 4735, 2514, 4794, 5006, 4487, 1104, 3012, 2026, 4717, 2812, 3377, 4706, 1380, 1086, 3172, 1602, 2156, 1559, 3501, 6398, 926, 2676, 590, 3138, 3108, 1741, 860, 2536, 6162, 2529, 1703, 1302, 3986, 1483, 1312, 796, 3196, 2459, 4052, 1468, 3068, 6269, 3914, 753, 6224, 4477, 6417, 3171, 6072, 3904, 3912, 3899, 5745, 1651, 2853, 3871, 3645, 1420, 3156, 3553, 5012, 2515, 3924, 1912, 2729, 1451, 5610, 1584, 1172, 3384, 1293, 2256, 1449, 6012, 1997, 3794, 3528, 1128, 1545, 4013, 4065, 1498, 732, 6222, 3060, 2018, 1560, 2028, 3278, 2275, 6041, 616, 3978, 4523, 1261, 5246, 3281, 2973, 3709, 1406, 6333, 2265, 4146, 4596, 2290, 5277, 4251, 580, 4612, 3093, 2857, 3189, 1764, 2060, 1751, 1273, 4864, 1040, 2066, 4539, 1158, 4404, 1227, 893, 2818, 6024, 2433, 3316, 4425, 4607, 2467, 2438, 4847, 1713, 3345, 4771, 4663, 1923, 2690, 1819, 5941, 755, 4416, 1359, 2001, 682, 1929, 1080, 3527, 3276, 901, 3157, 3413, 1284, 1915, 806, 2508, 2528, 2851, 3550, 5014, 1811, 1672, 1296, 3786, 1440, 3192, 4986, 5996, 4839, 1790, 4437, 1694, 4558, 2639, 836, 696, 3670, 719, 3459, 4018, 3366, 1131, 1905, 1218, 3296, 2665, 3700, 881, 3744, 907, 1494, 1317, 3925, 1585, 655, 2728, 912, 2533, 3850, 3631, 3533, 2738, 6872, 2978, 2119, 2878, 1134, 1787, 672, 3665, 4004, 4178, 4422, 3559, 5063, 3461, 5753, 3483, 4513, 2411, 4389, 2750, 1365, 4824, 620, 707, 1379, 1512, 3797, 2624, 1727, 1097, 717, 6262, 1280, 4950, 2986, 3508, 2963, 3358, 1489, 760, 3504, 1327, 4584, 4968, 4092, 4242, 596, 2015, 1337, 3843, 1438, 2543, 3491, 2888, 863, 4935, 3616, 1228, 3837, 3201, 2143, 4484, 2268, 4168, 3792, 2689, 2344, 1277, 3557, 1322, 3106, 3885, 1363, 842, 5808, 1681, 1467, 1732, 2465, 4381, 3091, 1355, 4135, 1771, 1852, 1115, 2198, 2234, 4420, 2388, 3713, 4164, 4074, 1222, 679, 3181, 4267, 3230, 3109, 1532, 917, 2439, 1872, 2019, 4031, 5942, 6233, 2002, 2263, 4452, 3002, 3597, 3886, 2819, 834, 3463, 4731, 3135, 1887, 4440, 3812, 4716, 1174, 2929, 2301, 2253, 1338, 3741, 4472, 1911, 2249, 2455, 2602, 2578, 2568, 4733, 964, 4867, 3438, 3723, 4774, 2594, 1514, 2627, 4217, 584, 574, 3224, 4763, 4027, 6716, 1032, 1529, 4421, 2511, 3579, 4254, 1426, 2418, 2516, 4411, 3893, 4503, 4995, 2421, 4678, 3755, 2491, 4532, 4597, 4108, 3092, 3382, 3877, 3337, 2832, 2998, 3629, 2149, 2972, 3067, 2316, 2993, 3636, 2517, 2962, 3617, 3124, 1864, 2987, 1580, 3987, 2276, 2926, 1381, 1027, 2103, 1930, 1041, 2892, 3791, 2545, 1065, 2360, 2184, 3206, 4577, 2572, 2871, 3485]

# COMMAND ----------

#display(pd['name'])
listEventId = []
index = 0
for line in sqlNotInId:
   index=index+1
print("Total UCM :",str(index))

# COMMAND ----------

pdDF = dbutils.fs.ls("file:/databricks/driver/datafileNMD/")

# COMMAND ----------

#display(pd['name'])
index = 0
for line in pdDF:
   index=index+1
print("Total NMD :",str(index))

# COMMAND ----------

pdDF = dbutils.fs.ls("dbfs:/RC/RC_EVENTS/ALL_FILES_RV3")
import pandas as pd

# COMMAND ----------

x= 489+541
print(x)

# COMMAND ----------

#display(pd['name'])
listEventId = []
index = 0
for line in pdDF:
  fullFile = line[1].split('.')
  tokenName = str(fullFile[0]).split('_')
  eventId = tokenName[3]
  listEventId.insert(index, eventId)
  print(line[1], "added ",eventId)
  index=index+1

# COMMAND ----------

print(listEventId)
print("index:",index
normalizedQTSDF = spark.createDataFrame(listEventId)

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/mnt/Exploratory/WCLD/11_BDLSTM/BILSTM_RCPredictiveMA_RV02_20190301_R2.zip file:/databricks/driver/

# COMMAND ----------

# MAGIC %sh
# MAGIC unzip BILSTM_RCPredictiveMA_RV02_20190301_R2.zip

# COMMAND ----------

# MAGIC %fs
# MAGIC cp dbfs:/RC/RC_EVENTS/ALL_FILES_RV3/BEWJ_10034_ID_642.csv dbfs:/mnt/Exploratory/WCLD/dataset

# COMMAND ----------

