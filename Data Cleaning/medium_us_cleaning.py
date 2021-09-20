import pandas as pd
import numpy as np

cols=['Age', 'Sex', 'Air velocity (m/s)', 'Air temperature (¡C)',
       'Radiant temperature (¡C)', 'Relative humidity (%)', 'Clo', 'Met',
       'Outdoor monthly air temperature (¡C)', 'Thermal sensation']

# Reading the raw data files
# Modify path_str accordingly
path_str = '/content/drive/My Drive/Colab Notebooks/Thermal sensation prediction(Murata)/LANGEVIN_DATA.txt'
f=open(path_str,'r')
text_us=f.readlines()

lens=[]
for i in range(len(text_us)):
  if(len(text_us[i].split())!=118):
    lens.append((i,len(text_us[i].split())))

dicts={}
for i in range(1,119):
  dicts[i]=[]

for i in range(len(text_us)):
  temp=text_us[i].split()
  for j in range(len(temp)):
    dicts[j+1].append(float(temp[j]))

df=pd.DataFrame(dicts)

# Using only significant cols
# Use Metadata file to identify the corresponding column numbers
data_us=df[[27,26,8,6,9,7,17,16,12,48]]

data_us.columns=cols

data_us=data_us.loc[data_us['Thermal sensation'].notnull()]
data_us=data_us.loc[data_us['Sex'].notnull()]



#Making the range from [-3,3] to[2,2]
data_us['Thermal sensation'] = data_us['Thermal sensation'].apply(lambda x: -2 if x <= -2 else x)
data_us['Thermal sensation'] = data_us['Thermal sensation'].apply(lambda x: 2 if x >= 2 else x)
#Rounding off the values to make it categorical in nature 
data_us['Thermal sensation'] = data_us['Thermal sensation'].apply(lambda x: np.round(x))

data_us = data_us.fillna(value=data_us.median())
data_us=data_us.drop_duplicates()
data_us.dropna(inplace=True)

data_us=data_us[data_us['Air velocity (m/s)']>0]
data_us=data_us[data_us['Air temperature (¡C)']>0]
data_us=data_us[data_us['Radiant temperature (¡C)']>0]

cols=['Air velocity (m/s)','Air temperature (¡C)','Radiant temperature (¡C)','Relative humidity (%)']  
for col in cols:
  data_us[col]=data_us[col].apply(lambda x: np.round(x,2))

from sklearn.cluster import DBSCAN

def DBSCAN_outlier_detection(data):
  outlier_detection=DBSCAN(min_samples=5,eps=3)
  clusters=outlier_detection.fit_predict(data)
  data['Clusters']=clusters
  data=data[data['Clusters']!=-1]
  data=data.drop(['Clusters'],axis=1)
  return(data)


data_us=DBSCAN_outlier_detection(data_us)
data_us.to_csv('Medium_US.csv')

