# Databricks notebook source
def displayImgDBFS(tfb_log_dir,dbfsImgPath, containsKey1, containsKey2,imgSizeX=1280,imgSizeY=640):
  dataFrameLogs = dbutils.fs.ls(tfb_log_dir)
  line = []
  htmlStr = "<table style='width:100%'>"
  
  # paths = dataFrameLogs[0]
  for path in dataFrameLogs:
    line = path

    if line[1].__contains__(containsKey1) and line[1].__contains__(containsKey2):
      filePath = dbfsImgPath + line[1]
      
      print (filePath)
      htmlStr = htmlStr + "<tr><img src='"+filePath+"' style='width:"+str(imgSizeX)+"px;height:"+str(imgSizeY)+"px;'></tr>"
      htmlStr = htmlStr+"<tr>"+filePath+"</tr>"

  htmlStr=htmlStr+"</tr></table>"
  return htmlStr

# COMMAND ----------


tfb_log_dir = "dbfs:/FileStore/images/RC_DATA_ANALYSIS/00_eda_101_201902010308"
dbfsImgPath = "files/images/RC_DATA_ANALYSIS/00_eda_101_201902010308/"
containsKey1 ="UCM_SEL_LINE"
containsKey2 ="LUBE_OIL_PRESS_"
htmpString = displayImgDBFS(tfb_log_dir,dbfsImgPath, containsKey1, containsKey2,1280,640)

# COMMAND ----------

displayHTML(htmpString)

# COMMAND ----------

# MAGIC %sql
# MAGIC select EVENT_ID
# MAGIC from rc_pi_data where BE_SD_TYPE = 'UCM'
# MAGIC and S0_LUBE_OIL_PRESS>80
# MAGIC and RUL = 420

# COMMAND ----------

# MAGIC %sql
# MAGIC select (RUL*-1) as IN_RUL,OBJECT_CODE, EVENT_ID, S0_LUBE_OIL_PRESS 
# MAGIC from rc_pi_data where BE_SD_TYPE = 'UCM'
# MAGIC and RUL between 1 and 420
# MAGIC and EVENT_ID in (1515,1572,295)
# MAGIC order by (RUL*-1)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from rc_sd_event_map where EVENT_ID = 1572

# COMMAND ----------

# MAGIC %sql
# MAGIC select RUL,OBJECT_CODE, EVENT_ID, S0_LUBE_OIL_PRESS 
# MAGIC from rc_pi_data where BE_SD_TYPE = 'UCM'
# MAGIC and RUL between 100 and 720
# MAGIC and EVENT_ID in (1515,1527,295)

# COMMAND ----------

# %sql
# CREATE INDEX rc_pi_data_idx1 ON TABLE rc_pi_data (RUL,EVENT_ID,BE_SD_TYPE) AS 'COMPACT';

# COMMAND ----------

# %sql
# CREATE DATASKIPPING INDEX ON [TABLE] rc_pi_data
#     FOR COLUMNS (RUL,EVENT_ID,BE_SD_TYPE)

# COMMAND ----------

#/FileStore/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM/11_M9_LSTM_model_verify.png

tfb_log_dir = "dbfs:/FileStore/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM/"
dbfsImgPath = "files/images/RC_DATA_ANALYSIS/MINI_BATCH_LSTM/"
containsKey1 ="_model_verify"
containsKey2 ="_model_verify.png"
htmpString = displayImgDBFS(tfb_log_dir,dbfsImgPath, containsKey1, containsKey2,840,420)

# COMMAND ----------

displayHTML(htmpString)

# COMMAND ----------

