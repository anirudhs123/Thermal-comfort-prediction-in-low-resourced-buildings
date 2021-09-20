import pandas as pd
import numpy as np

# Loading the Raw datafile
# Modify pathstr accordingly
path_str = '/content/drive/MyDrive/Colab Notebooks/Thermal sensation prediction(Murata)/ashrae_db2.01.csv'
data_ash=pd.read_csv(path_str,encoding='ISO-8859-1')

#Removing all the other data not having AC as thier Cooling stratergy
data1=data_ash[data_ash['Cooling startegy_building level']=='Air Conditioned']
data_ash=pd.concat([data1,data2,data3],axis=0)

#Making the range from [-3,3] to[2,2]
data_ash['Thermal sensation'] = data_ash['Thermal sensation'].apply(lambda x: -2 if x <= -2 else x)
data_ash['Thermal sensation'] = data_ash['Thermal sensation'].apply(lambda x: 2 if x >= 2 else x)
#Rounding off the values to make it categorical in nature 
data_ash['Thermal sensation'] = data_ash['Thermal sensation'].apply(lambda x: np.round(x))


data_ash=data_ash.loc[data_ash['Thermal sensation'].notnull()]
data_ash=data_ash.loc[data_ash['Sex'].notnull()]

#data_ash['Cooling startegy_building level'].value_counts()
data_ash['Thermal sensation'].value_counts()

data_ash=data_ash[['Age','Sex','Air velocity (m/s)','Air temperature (¡C)','Radiant temperature (¡C)','Relative humidity (%)','Clo', 'Met','Outdoor monthly air temperature (¡C)','Thermal sensation']]

data_ash=data_ash.drop_duplicates()

def fun(str1):
  if(str1=='Female'):
    return(2.0)
  elif(str1=='Male'):
    return(1.0)

    
data_ash['Sex']=data_ash['Sex'].apply(lambda x: fun(x))

data_ash.columns

data_ash=data_ash[data_ash['Air velocity (m/s)']>0]
data_ash=data_ash[data_ash['Air temperature (¡C)']>0]
data_ash=data_ash[data_ash['Radiant temperature (¡C)']>0]

data_ash['Clo'].describe()

data_ash=data_ash.fillna(data_ash.median())

data_ash=data_ash.drop_duplicates()
data_ash=data_ash.dropna()



from sklearn.cluster import DBSCAN

def DBSCAN_outlier_detection(data):
  outlier_detection=DBSCAN(min_samples=5,eps=3)
  clusters=outlier_detection.fit_predict(data)
  data['Clusters']=clusters
  data=data[data['Clusters']!=-1]
  data=data.drop(['Clusters'],axis=1)
  return(data)

data_ash=DBSCAN_outlier_detection(data_ash)

data_ash.to_csv('Ashrae.csv')