# Databricks notebook source
# MAGIC %md # Notebook #1
# MAGIC 
# MAGIC In this notebook, we show how to import, clean and create features relevent for predictive maintenance data using PySpark. This notebook uses Spark **2.0.2** and Python Python **2.7.5**. The API documentation for that version can be found [here](https://spark.apache.org/docs/2.0.2/api/python/index.html).
# MAGIC 
# MAGIC ## Outline
# MAGIC 
# MAGIC - [Import Data](#Import-Data)
# MAGIC - [Data Exploration & Cleansing](#Data-Exploration-&-Cleansing)
# MAGIC - [Feature Engineering](#Feature-Engineering)
# MAGIC - [Save Result](#Save-Result)

# COMMAND ----------

import subprocess
import sys
import os
import re
import time
import atexit
import seaborn as sns
import matplotlib.pyplot as plt

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import concat, col, udf, lag, date_add, explode, lit, unix_timestamp
from pyspark.sql.functions import month, weekofyear, dayofmonth
from pyspark.sql.types import *
from pyspark.sql.types import DateType
from pyspark.sql.dataframe import *
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.ml.classification import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, VectorIndexer
from pyspark.ml.feature import StandardScaler, PCA, RFormula
from pyspark.ml import Pipeline, PipelineModel

start_time = time.time()


# COMMAND ----------

# MAGIC %md ## Import Data

# COMMAND ----------

# MAGIC %md #### We initially encountered some issue while reading the data
# MAGIC -  For Spark 2.0 and above, we can use the sqlContext.read.csv to import data directly into the Spark context. The data import seems to work fine and you can also perform some data transformation without any problem. However, when we tried to show the top n rows of the entire data (e.g. data.show(3)) or do some data manipulation on certain columns downstream, we encountered error of “Null Pointer” or “NumberFormatException: null”.
# MAGIC -  In our case, it was because for some numeric columns with missing records containing "", Spark still recognizes those column as numeric but somehow cannot parse them correctly. Hopefully the future version of Spark could handle such issue more intelligently. 
# MAGIC -  We fixed the problem by reformating the data before loading into Spark context as csv format.
# MAGIC ```bash
# MAGIC  cat data.csv | tr -d "\"\"" > sampledata.csv 
# MAGIC ```

# COMMAND ----------

# MAGIC %md #### Import data from Azure Blob Storage
# MAGIC The input data is hosted on a publicly accessible Azure Blob Storage container and can be downloaded from [here](https://pysparksampledata.blob.core.windows.net/sampledata/sampledata.csv).
# MAGIC 
# MAGIC To learn how to grant read-only access to Azure storage containers or blobs without sharing your account key and without requiring a shared access signature (SAS), please follow the instructions [here](https://docs.microsoft.com/en-us/azure/storage/storage-manage-access-to-resources).  

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -O https://pysparksampledata.blob.core.windows.net/sampledata/sampledata.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -al
# MAGIC pwd

# COMMAND ----------

# MAGIC %fs
# MAGIC cp file:/databricks/driver/sampledata.csv dbfs:/RC/example

# COMMAND ----------

# Import data from Azure Blob Storage
#dataFile = "wasb://sampledata@pysparksampledata.blob.core.windows.net/sampledata.csv"
dataFileSep = ','
# Import data from the home directory on your machine 
dataFile = 'dbfs:/RC/example/sampledata.csv'
df = sqlContext.read.csv(dataFile, header=True, sep=dataFileSep, inferSchema=True, nanValue="", mode='PERMISSIVE')

# COMMAND ----------

# MAGIC %md ## Data Exploration & Cleansing

# COMMAND ----------

# MAGIC %md First, let's look at the dataset dimension and data schema.

# COMMAND ----------

# check the dimensions of the data
df.count(), len(df.columns)

# COMMAND ----------

# check whether the issue of df.show() is fixed
df.show(1)

# COMMAND ----------

# check data schema
df.dtypes

# COMMAND ----------

# MAGIC %md #### Explanations on the data schema:
# MAGIC * ***DeviceID***: machine identifier
# MAGIC * ***Date***: the day when that row of data was collected for that machine
# MAGIC * ***Categorical_1 to 4***: some categorical features about the machine
# MAGIC * ***Problem_Type_1 to 4***: the total number of times Problem type 1 (2, 3, 4) occured on that day for that machine
# MAGIC * ***Usage_Count_1 (2)***: the total number of times that machine had been used on that day for purpose type 1 or 2
# MAGIC * ***Warning_xxx***: the total number of Warning type_xxx occured for that machine on that day
# MAGIC * ***Error_Count_1 to 8***: the total number of times Error type 1 (to 8) occured on that day for that machine
# MAGIC * ***Fault_Code_Type_1 to 4***: fault code type 1 (2, 3, 4) occured on that day for that machine
# MAGIC * ***Problemreported***: prediction target column whether or not there is a machine problem on that day

# COMMAND ----------

# MAGIC %md #### As part of the data cleansing process, we standardized all the column names to lower case and replaced all the symbols with underscore. We also removed any duplicated records.

# COMMAND ----------

#--------------------------------------- initial data cleansing ---------------------------------------------#

# standardize the column names
def StandardizeNames(df):
    l = df.columns
    cols = [c.replace(' ','_').
              replace('[.]','_').
              replace('.','_').
              replace('[[:punct:]]','_').
              lower() for c in l]
    return df.toDF(*cols)
df = StandardizeNames(df)

# remove duplicated rows based on deviceid and date
df = df.dropDuplicates(['deviceid', 'date'])

# remove rows with missing deviceid, date
df = df.dropna(how='any', subset=['deviceid', 'date'])

df.select('deviceid','date').show(3)


# COMMAND ----------

# MAGIC %md #### Define groups of features -- date, categorical, numeric

# COMMAND ----------

#------------------------------------------- Define groups of features -----------------------------------------#

features_datetime = ['date']

features_categorical = ['deviceid','Categorical_1','Categorical_2','Categorical_3','Categorical_4',
                        'fault_code_type_1','fault_code_type_2',
                        'fault_code_type_3','fault_code_type_4',
                        'problemreported']

features_numeric = list(set(df.columns) -set(features_datetime)-set(features_categorical))


# COMMAND ----------

# MAGIC %md #### Handling missing data

# COMMAND ----------

print(df['fault_code_type_3',].head(3))
# there are some missing values, we need to handle in the subsequent steps


# COMMAND ----------

# handle missing values
df = df.fillna(0, subset=features_numeric)
df = df.fillna("Unknown", subset=features_categorical)

# check the results
print(df['fault_code_type_3',].head(3))


# COMMAND ----------

# MAGIC %md #### For data exploration part, people usually would like to visualize the distribution of certain columns or the interation among columns. Here, we hand picked some columns to demonstrate how to do some basic visualizations.

# COMMAND ----------

#------------------------------------ data exploration and visualization ------------------------------------#

# Register dataframe as a temp table in SQL context
df.createOrReplaceTempView("df1")

sqlStatement = """
    SELECT problem_type_1, problem_type_2, problem_type_3, problem_type_4,
    error_count_1, error_count_2, error_count_3, error_count_4, 
    error_count_5, error_count_6, error_count_7, error_count_8, problemreported
    FROM df1
"""
plotdata = spark.sql(sqlStatement).toPandas();


%matplotlib inline

# show histogram distribution of some features
ax1 = plotdata[['problem_type_1']].plot(kind='hist', bins=5, facecolor='blue')
ax1.set_title('problem_type_1 distribution')
ax1.set_xlabel('number of problem_type_1 per day'); ax1.set_ylabel('Counts');
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()

ax1 = plotdata[['problem_type_2']].plot(kind='hist', bins=5, facecolor='blue')
ax1.set_title('problem_type_2 distribution')
ax1.set_xlabel('number of problem_type_2 per day'); ax1.set_ylabel('Counts');
plt.figure(figsize=(4,4)); plt.suptitle(''); plt.show()


# show correlation matrix heatmap to explore some potential interesting patterns
corr = plotdata.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
display()

# COMMAND ----------

# MAGIC %md ## Feature Engineering
# MAGIC In the remaining part of the Notebook #1, we will demonstrate how to generate new features for this kind of use case. It is definitely not meant to be a comprehensive list.

# COMMAND ----------

# MAGIC %md In the following cell, we created some time features, calculated the total number of warning_type1 (type2) occured for a macine on a particular day. We also identified some data quality issue that some event counts had negative values. 

# COMMAND ----------

# Extract some time features from "date" column
df = df.withColumn('month', month(df['date']))
df = df.withColumn('weekofyear', weekofyear(df['date']))
df = df.withColumn('dayofmonth', dayofmonth(df['date']))


# warning related raw features
warning_type1_features = list(s for s in df.columns if "warning_1_" in s) 
                            
warning_type2_features = list(s for s in df.columns if "warning_2_" in s)

warning_all = warning_type1_features + warning_type2_features

# total count of all type1 warnings each day each device
df = df.withColumn('warn_type1_total', sum(df[col_n] for col_n in warning_type1_features))
# total count of all type2 warnings each day each device
df = df.withColumn('warn_type2_total', sum(df[col_n] for col_n in warning_type2_features))

print(df['warn_type1_total',].head(3))
print(df['warn_type2_total',].head(3))


# COMMAND ----------

# We realized that the warning counts have negative values
# Replace all the negative values with 0

def negative_replace(num):
    if num < 0: return 0
    else: return num
    
negative_replace_Udf = udf(negative_replace, IntegerType())

m = warning_type1_features + warning_type2_features
for col_n in m:
    df = df.withColumn(col_n, negative_replace_Udf(df[col_n]))

# Then we have to re-calculate the total warnings again 
df = df.withColumn('warn_type1_total', sum(df[col_n] for col_n in warning_type1_features))
df = df.withColumn('warn_type2_total', sum(df[col_n] for col_n in warning_type2_features))

print(df['warn_type1_total',].head(3))
print(df['warn_type2_total',].head(3))


# COMMAND ----------

# MAGIC %md #### Variables "categorical_1 to 4" are integer type but in fact they are categorical features. In the following cell, we binned those variables and created four new columns.

# COMMAND ----------

# Note: we can also use SparkSQL for this binning task

def Cat1(num):
    if num <= 10: return '0-10'
    elif 10 < num and num <= 20: return '11-20'
    elif 20 < num and num <= 30: return '21-30'
    elif 30 < num and num <= 40: return '31-40'
    else: return 'morethan40'
cat1Udf = udf(Cat1, StringType())
df = df.withColumn("cat1", cat1Udf('categorical_1'))


def Cat2(num):
    if num <= 2000: return '0-2000'
    elif 2000 < num and num <= 3000: return '2000-3000'
    elif 3000 < num and num <= 4000: return '3000-4000'
    elif 4000 < num and num <= 5000: return '4000-5000'
    elif 5000 < num and num <= 6000: return '5000-6000'
    else: return 'morethan6000'
cat2Udf = udf(Cat2, StringType())
df = df.withColumn("cat2", cat2Udf('categorical_2'))


def Cat3(num):
    if num <= 200: return '0-200'
    elif 200 < num and num <= 400: return '200-400'
    elif 400 < num and num <= 600: return '400-600'
    elif 600 < num and num <= 800: return '600-800'
    else: return 'morethan800'
cat3Udf = udf(Cat3, StringType())
df = df.withColumn("cat3", cat3Udf('categorical_3'))


def Cat4(num):
    if num <= 5000: return '0-5000'
    elif 5000 < num and num <= 10000: return '5000-10000'
    elif 10000 < num and num <= 15000: return '10000-15000'
    elif 15000 < num and num <= 20000: return '15000-20000'
    else: return 'morethan20000'
cat4Udf = udf(Cat4, StringType())
df = df.withColumn("cat4", cat4Udf('categorical_4'))


print(df.select('cat1').distinct().rdd.map(lambda r: r[0]).collect())
print(df.select('cat2').distinct().rdd.map(lambda r: r[0]).collect())
print(df.select('cat3').distinct().rdd.map(lambda r: r[0]).collect())
print(df.select('cat4').distinct().rdd.map(lambda r: r[0]).collect())


# COMMAND ----------

# MAGIC %md #### For variables "fault_code_type_1 to 4", if it is "Unknown" that means there is "0" fault code reported on that day for that machine, otherwise the count of fault code type 1 (2, 3, or 4) is 1.

# COMMAND ----------

df = df.withColumn("fault_code_type_1_count",F.when(df.fault_code_type_1!= "Unknown", 1).otherwise(0))\
       .withColumn("fault_code_type_2_count",F.when(df.fault_code_type_2!= "Unknown", 1).otherwise(0))\
       .withColumn("fault_code_type_3_count",F.when(df.fault_code_type_3!= "Unknown", 1).otherwise(0))\
       .withColumn("fault_code_type_4_count",F.when(df.fault_code_type_4!= "Unknown", 1).otherwise(0))

df.groupby('fault_code_type_1_count').count().show()
df.groupby('fault_code_type_2_count').count().show()
df.groupby('fault_code_type_3_count').count().show()
df.groupby('fault_code_type_4_count').count().show()


# COMMAND ----------

# MAGIC %md #### Feature engineering performance related features
# MAGIC We first select 8 raw performance features to be normalized and then select 2 normalizers.  
# MAGIC The idea behind this normalization is that device with more problem/error/fault reported might simply because it is used more frequently. Therefore, we need to normalize the problem counts by the corresponding usage counts.

# COMMAND ----------

# First, select the 8 raw performance features to be normalized
performance_normal_raw = ['problem_type_1','problem_type_2','problem_type_3','problem_type_4',
                          'fault_code_type_1_count','fault_code_type_2_count',
                          'fault_code_type_3_count', 'fault_code_type_4_count']

# Then, select 2 normalizers
performance_normalizer = ['usage_count_1','usage_count_2']

# Normalize performance_normal_raw by "usage_count_1"
df = df.withColumn("problem_type_1_per_usage1", F.when(df.usage_count_1==0,0).otherwise(df.problem_type_1/df.usage_count_1))\
       .withColumn("problem_type_2_per_usage1",F.when(df.usage_count_1==0,0).otherwise(df.problem_type_2/df.usage_count_1))\
       .withColumn("problem_type_3_per_usage1",F.when(df.usage_count_1==0,0).otherwise(df.problem_type_3/df.usage_count_1))\
       .withColumn("problem_type_4_per_usage1",F.when(df.usage_count_1==0,0).otherwise(df.problem_type_4/df.usage_count_1))\
       .withColumn("fault_code_type_1_count_per_usage1",F.when(df.usage_count_1==0,0).otherwise(df.fault_code_type_1_count/df.usage_count_1))\
       .withColumn("fault_code_type_2_count_per_usage1",F.when(df.usage_count_1==0,0).otherwise(df.fault_code_type_2_count/df.usage_count_1))\
       .withColumn("fault_code_type_3_count_per_usage1",F.when(df.usage_count_1==0,0).otherwise(df.fault_code_type_3_count/df.usage_count_1))\
       .withColumn("fault_code_type_4_count_per_usage1",F.when(df.usage_count_1==0,0).otherwise(df.fault_code_type_4_count/df.usage_count_1))

# Normalize performance_normal_raw by "usage_count_2"
df = df.withColumn("problem_type_1_per_usage2", F.when(df.usage_count_2==0,0).otherwise(df.problem_type_1/df.usage_count_2))\
       .withColumn("problem_type_2_per_usage2",F.when(df.usage_count_2==0,0).otherwise(df.problem_type_2/df.usage_count_2))\
       .withColumn("problem_type_3_per_usage2",F.when(df.usage_count_2==0,0).otherwise(df.problem_type_3/df.usage_count_2))\
       .withColumn("problem_type_4_per_usage2",F.when(df.usage_count_2==0,0).otherwise(df.problem_type_4/df.usage_count_2))\
       .withColumn("fault_code_type_1_count_per_usage2",F.when(df.usage_count_2==0,0).otherwise(df.fault_code_type_1_count/df.usage_count_2))\
       .withColumn("fault_code_type_2_count_per_usage2",F.when(df.usage_count_2==0,0).otherwise(df.fault_code_type_2_count/df.usage_count_2))\
       .withColumn("fault_code_type_3_count_per_usage2",F.when(df.usage_count_2==0,0).otherwise(df.fault_code_type_3_count/df.usage_count_2))\
       .withColumn("fault_code_type_4_count_per_usage2",F.when(df.usage_count_2==0,0).otherwise(df.fault_code_type_4_count/df.usage_count_2))


# COMMAND ----------

# MAGIC %md #### Similar to what we did for "categorical_1 to 4", in the following cell we binned performance related features and created new categorical features. 

# COMMAND ----------

# Define the list of performance related features which we would like to perform binning
c_names = ['problem_type_1', 'problem_type_3', 'problem_type_4',
           'problem_type_1_per_usage1','problem_type_2_per_usage1','problem_type_3_per_usage1','problem_type_4_per_usage1',
           'problem_type_1_per_usage2','problem_type_2_per_usage2','problem_type_3_per_usage2','problem_type_4_per_usage2',
           'fault_code_type_1_count', 'fault_code_type_2_count', 'fault_code_type_3_count', 'fault_code_type_4_count',                          
           'fault_code_type_1_count_per_usage1','fault_code_type_2_count_per_usage1',
           'fault_code_type_3_count_per_usage1', 'fault_code_type_4_count_per_usage1',
           'fault_code_type_1_count_per_usage2','fault_code_type_2_count_per_usage2',
           'fault_code_type_3_count_per_usage2', 'fault_code_type_4_count_per_usage2']

# Bin size ('0','1','>1') for most of the performance features because majority of the values fall into the range of 0 to slightly more than 1.
def performanceCat(num):
    if num == 0: return '0'
    elif num ==1: return '1'
    else: return '>1'
    
performanceCatUdf = udf(performanceCat, StringType())
for col_n in c_names:
    df = df.withColumn(col_n+'_category',performanceCatUdf(df[col_n]))

# Use different bin for "problem_type_2" because we saw a larger spread of the values
def problem_type_2_Cat(num):
    if num == 0: return '0'
    elif 0 < num and num <= 5: return '1-5'
    elif 5 < num and num <= 10: return '6-10'
    else: return '>10'

problem_type_2_CatUdf = udf(problem_type_2_Cat, StringType())
df = df.withColumn('problem_type_2_category',problem_type_2_CatUdf(df['problem_type_2']))


print(df.select('problem_type_1_category').distinct().rdd.map(lambda r: r[0]).collect())
print(df.select('problem_type_2_category').distinct().rdd.map(lambda r: r[0]).collect())


# COMMAND ----------

# MAGIC %md #### One hot encode some categotical features

# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC # Define the list of categorical features
# MAGIC 
# MAGIC catVarNames = ['problem_type_1_category', 'problem_type_2_category',
# MAGIC                'problem_type_3_category', 'problem_type_4_category',
# MAGIC                'problem_type_1_per_usage1_category', 'problem_type_2_per_usage1_category',
# MAGIC                'problem_type_3_per_usage1_category', 'problem_type_4_per_usage1_category',
# MAGIC                'problem_type_1_per_usage2_category', 'problem_type_2_per_usage2_category',
# MAGIC                'problem_type_3_per_usage2_category', 'problem_type_4_per_usage2_category',
# MAGIC                'fault_code_type_1_count_category', 'fault_code_type_2_count_category',
# MAGIC                'fault_code_type_3_count_category', 'fault_code_type_4_count_category',
# MAGIC                'fault_code_type_1_count_per_usage1_category', 'fault_code_type_2_count_per_usage1_category',
# MAGIC                'fault_code_type_3_count_per_usage1_category', 'fault_code_type_4_count_per_usage1_category',
# MAGIC                'fault_code_type_1_count_per_usage2_category', 'fault_code_type_2_count_per_usage2_category',
# MAGIC                'fault_code_type_3_count_per_usage2_category', 'fault_code_type_4_count_per_usage2_category',
# MAGIC                'cat1','cat2','cat3','cat4']
# MAGIC     
# MAGIC     
# MAGIC sIndexers = [StringIndexer(inputCol=x, outputCol=x + '_indexed') for x in catVarNames]
# MAGIC 
# MAGIC df_cat = Pipeline(stages=sIndexers).fit(df).transform(df)
# MAGIC 
# MAGIC # Remove columns with only 1 level (compute variances of columns)
# MAGIC catColVariance = df_cat.select(
# MAGIC     *(F.variance(df_cat[c]).alias(c + '_sd') for c in [cv + '_indexed' for cv in catVarNames]))
# MAGIC catColVariance = catColVariance.rdd.flatMap(lambda x: x).collect()
# MAGIC catVarNames = [catVarNames[k] for k in [i for i, v in enumerate(catColVariance) if v != 0]]
# MAGIC 
# MAGIC # Encode
# MAGIC ohEncoders = [OneHotEncoder(inputCol=x + '_indexed', outputCol=x + '_encoded')
# MAGIC               for x in catVarNames]
# MAGIC ohPipelineModel = Pipeline(stages=ohEncoders).fit(df_cat)
# MAGIC df_cat = ohPipelineModel.transform(df_cat)
# MAGIC 
# MAGIC drop_list = [col_n for col_n in df_cat.columns if 'indexed' in col_n]
# MAGIC df = df_cat.select([column for column in df_cat.columns if column not in drop_list])
# MAGIC 
# MAGIC print(df['problem_type_1_category_encoded',].head(3))

# COMMAND ----------

# MAGIC %md #### Use PCA to reduce number of features
# MAGIC In Notebook #2, we will perform a series of rolling computation for various features, time windows and aggregated statistics. This process is very computational expensive and therefore we need to first reduce the feature list.  
# MAGIC In the dataset, there are many warning related features and most of them have value of 0 so quite sparse. We can group or find correlations among those warning features, reduce the feature space for downstream work.

# COMMAND ----------

## check the number of warning related features
len([col_n for col_n in df.columns if 'warning' in col_n])


# COMMAND ----------

# MAGIC %%time
# MAGIC 
# MAGIC #----------------------------- PCA feature grouping on warning related features --------------------------#
# MAGIC 
# MAGIC df = df.withColumn("key", concat(df.deviceid,lit("_"),df.date))
# MAGIC 
# MAGIC # step 1
# MAGIC # Use RFormula to create the feature vector
# MAGIC formula = RFormula(formula = "~" + "+".join(warning_all))
# MAGIC output = formula.fit(df).transform(df).select("key","features") 
# MAGIC 
# MAGIC 
# MAGIC # step 2
# MAGIC # Before PCA, we need to standardize the features, it is very important...
# MAGIC # Note that StandardScaler does not work for sparse vector unless withMean=false
# MAGIC # OR we can convert sparse vector to dense vector first using toArray
# MAGIC scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
# MAGIC                         withStd=True, withMean=False)
# MAGIC 
# MAGIC # Compute summary statistics by fitting the StandardScaler
# MAGIC scalerModel = scaler.fit(output)
# MAGIC 
# MAGIC # Normalize each feature to have unit standard deviation.
# MAGIC scaledData = scalerModel.transform(output)
# MAGIC 
# MAGIC 
# MAGIC # step 3
# MAGIC pca = PCA(k=20, inputCol="scaledFeatures", outputCol="pcaFeatures")
# MAGIC model = pca.fit(scaledData)
# MAGIC result = model.transform(scaledData).select("key","pcaFeatures")
# MAGIC 
# MAGIC # to check how much variance explained by each component
# MAGIC print(model.explainedVariance)  
# MAGIC 
# MAGIC 
# MAGIC # step 4
# MAGIC # convert pca result, a vector column, to mulitple columns
# MAGIC # The reason why we did this was because later on we need to use those columns to generate more features (rolling compute) 
# MAGIC def extract(row):
# MAGIC     return (row.key, ) + tuple(float(x) for x in row.pcaFeatures.values)
# MAGIC 
# MAGIC pca_outcome = result.rdd.map(extract).toDF(["key"])
# MAGIC 
# MAGIC # rename columns of pca_outcome
# MAGIC oldColumns = pca_outcome.schema.names
# MAGIC 
# MAGIC newColumns = ["key", 
# MAGIC               "pca_1_warn","pca_2_warn","pca_3_warn","pca_4_warn","pca_5_warn",
# MAGIC               "pca_6_warn","pca_7_warn","pca_8_warn","pca_9_warn","pca_10_warn",
# MAGIC               "pca_11_warn","pca_12_warn","pca_13_warn","pca_14_warn","pca_15_warn",
# MAGIC               "pca_16_warn","pca_17_warn","pca_18_warn","pca_19_warn","pca_20_warn",
# MAGIC              ]
# MAGIC 
# MAGIC pca_result = reduce(lambda pca_outcome, idx: pca_outcome.withColumnRenamed(oldColumns[idx], newColumns[idx]), \
# MAGIC                                         xrange(len(oldColumns)), pca_outcome)
# MAGIC 
# MAGIC df = df.join(pca_result, 'key', 'inner')
# MAGIC 
# MAGIC print(df['pca_1_warn',].head(3))
# MAGIC 
# MAGIC warning_drop_list = [col_n for col_n in df.columns if 'warning_' in col_n]
# MAGIC df = df.select([column for column in df.columns if column not in warning_drop_list])

# COMMAND ----------

# I would like to visualize the relationship among the 20 pca components

# Register dataframe as a temp table in SQL context
df.createOrReplaceTempView("df2")

sqlStatement2 = """
    SELECT pca_1_warn, pca_2_warn, pca_3_warn, pca_4_warn, pca_5_warn, 
    pca_6_warn, pca_7_warn, pca_8_warn, pca_9_warn, pca_10_warn,
    pca_11_warn, pca_12_warn, pca_13_warn, pca_14_warn, pca_15_warn, 
    pca_16_warn, pca_17_warn, pca_18_warn, pca_19_warn, pca_20_warn
    FROM df2
"""
plotdata2 = spark.sql(sqlStatement2).toPandas();


%matplotlib inline
# show correlation matrix heatmap to explore some potential interesting patterns
corr = plotdata2.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# From the plot we can see the 20 pca components do not overlap too much which is expected


# COMMAND ----------

# MAGIC %md ## Save Result
# MAGIC 
# MAGIC Due to the lazy compute of Spark, it is usually more efficient to break down the workload into chunks and materialize the intermediate results. For example, we divided the tutorial into three notebooks, the result from Notebook #1 would be used as input data for Notebook #2. 

# COMMAND ----------

# MAGIC %%time
# MAGIC /mnt/Exploratory/WCLD/BetaProject
# MAGIC df.write.mode('overwrite').parquet('/mnt/Exploratory/WCLD/BetaProject/notebook1_result.parquet')

# COMMAND ----------

