import pandas as pd
import numpy as np

# Loading data from local disk
# Modify path of the data accordingly

path_str = '/content/drive/My Drive/Colab Notebooks/Thermal sensation prediction(Murata)/FinalDataset_2019-04-30.csv'
data=pd.read_csv(path_str,encoding= 'unicode_escape')

#Index taken from the column description file
index_to_choose_for_cols=[45,44,123,124,120,113,112,40]
cols=data.columns
cols_chosen=[]
for i in index_to_choose_for_cols:
    cols_chosen.append(cols[i-1])

data=data[cols_chosen]
cols=['Age', 'Sex', 'Air velocity (m/s)', 'Air temperature (¡C)',
       'Radiant temperature (¡C)', 'Relative humidity (%)',
       'Outdoor monthly air temperature (¡C)', 'Thermal sensation']

data.columns=cols

data=data[data['Age'] !=7]
data=data[data['Sex']!=4]
data=data[data['Sex']!=3]
data=data[data['Thermal sensation'].notna()]

#They have used 7 classes with option for slightly cold and slightly warm extra options
#Combining them into the warm and cold categories
#Matchin dict 
matching_dict={1:-2,2:-1,3:-1,4:0,5:0,6:1,7:2}
def change(x):
  return(matching_dict[x])
data['Thermal sensation']=data['Thermal sensation'].apply(lambda x:change(x))



#Imputing the missing values
for col in data.columns:
  data[col]=data[col].fillna(data[col].median())


data=data[data['Air velocity (m/s)']>=0]
data=data[data['Air temperature (¡C)']>0]
data=data[data['Radiant temperature (¡C)']>0]

from sklearn.cluster import DBSCAN

def DBSCAN_outlier_detection(data):
  outlier_detection=DBSCAN(min_samples=5,eps=3)
  clusters=outlier_detection.fit_predict(data)
  data['Clusters']=clusters
  data=data[data['Clusters']!=-1]
  data=data.drop(['Clusters'],axis=1)
  return(data)

data=DBSCAN_outlier_detection(data)
data=data.drop_duplicates()
data.to_csv('Scales.csv')

