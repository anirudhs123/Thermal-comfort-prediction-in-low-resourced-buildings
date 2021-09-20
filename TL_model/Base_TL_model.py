import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn import *

from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.utils import resample

from keras.models import Sequential
from keras.layers import Dense,Dropout,ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras.layers import Embedding,Conv1D,LSTM,Input,TimeDistributed,SpatialDropout1D,Flatten,Dropout
from keras.models import Model

from sklearn.metrics import *

from tensorflow.keras import models, layers, optimizers,regularizers,preprocessing as kprocessing
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping,ModelCheckpoint

#For plotting
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

# Read the data from local disk
# Change the paths accordingly 
data_ash=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Thermal sensation prediction(Murata)/Ashrae_data_fin.csv')
data_scales=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Thermal sensation prediction(Murata)/Scales_data_fin.csv')
data_us=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Thermal sensation prediction(Murata)/Medium_US_data_fin.csv')

# Changing from [-2,2] to [0,4] 
data_ash['Thermal sensation']=data_ash['Thermal sensation'].apply(lambda x: x+2)
data_scales['Thermal sensation']=data_scales['Thermal sensation'].apply(lambda x: x+2)
data_us['Thermal sensation']=data_us['Thermal sensation'].apply(lambda x: x+2)

# Encoding age into categories 
def age_encode(x):
  if(x==1):
    return(18)
  elif(x==2):
    return(23)  
  elif(x==3):
    return(28) 
  elif(x==4):
    return(33) 
  elif(x==5):
    return(38) 
  elif(x==6):
    return(50)    
data_scales['Age']=data_scales['Age'].apply(lambda x: age_encode(x))

#Downsampling the Data
def Down_sampling_data(df):
  df_0 = df[df['Thermal sensation']==0]
  df_1 = df[df['Thermal sensation']==1]
  df_2 = df[df['Thermal sensation']==2]
  df_3 = df[df['Thermal sensation']==3]
  df_4 = df[df['Thermal sensation']==4]

  # Downsample majority class
  lens=[len(df_0),len(df_1),len(df_2),len(df_3),len(df_4)]
  max_ind=np.argmax(lens)
  min_ind=np.argmin(lens)
  dfs=[df_0,df_1,df_2,df_3,df_4]
  df_minority=dfs[min_ind]
  dfs_majority_downsampled=[]
  for i in range(5):
    if(i!=min_ind): 
      df_majority=dfs[i]
      df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=len(df_minority),# to match minority class
                                 random_state=123) # reproducible results
      dfs_majority_downsampled.append(df_majority_downsampled)                           
 
  # Combine minority class with downsampled majority class
  df_downsampled = pd.concat(dfs_majority_downsampled+[df_minority])
  return(df_downsampled)

  # Upsample minority classes
def Up_sampling_data(df):
  df_0 = df[df['Thermal sensation']==0]
  df_1 = df[df['Thermal sensation']==1]
  df_2 = df[df['Thermal sensation']==2]
  df_3 = df[df['Thermal sensation']==3]
  df_4 = df[df['Thermal sensation']==4]
  # Upsample minority class
  lens=[len(df_0),len(df_1),len(df_2),len(df_3),len(df_4)]
  max_ind=np.argmax(lens)
  min_ind=np.argmin(lens)
  dfs=[df_0,df_1,df_2,df_3,df_4]
  df_majority=dfs[max_ind]
  dfs_minority_upsampled=[]
  for i in range(5):
    if(i!=max_ind): 
      df_minority=dfs[i]
      df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123)
      
      dfs_minority_upsampled.append(df_minority_upsampled)                           
 
  # Combine minority class with downsampled majority class
  df_upsampled = pd.concat(dfs_minority_upsampled+[df_majority])
  return(df_upsampled)

#Downsmapling datasets
data_ash=Down_sampling_data(data_ash)
data_us=Down_sampling_data(data_us)
data_scales=Down_sampling_data(data_scales)

 # Baseline models - Comparative study
SVM_lin_model=SVC(class_weight='balanced',kernel='linear')
SVM_poly_model=SVC(class_weight='balanced',kernel='poly')
SVM_radial_model=SVC(class_weight='balanced',kernel='rbf')

NB_model=GaussianNB()
KNN_model=KNeighborsClassifier(n_neighbors=5,weights='distance')

RF_model=RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',max_depth=10)
DT_model=DecisionTreeClassifier(max_depth=10,class_weight='balanced')
Adaboost_model=AdaBoostClassifier(n_estimators=100,random_state=1)

# Dropping Unwanted cols for different feature sets
X1=data_ash.drop(['Thermal sensation','Clo','Met'],axis=1)
y1=data_ash['Thermal sensation']

X2=data_us.drop(['Thermal sensation','Clo','Met'],axis=1)
y2=data_us['Thermal sensation']

X3=data_scales.drop(['Thermal sensation'],axis=1)
y3=data_scales['Thermal sensation']

# Data Preprocessing Pipeline
def Standard_Scaler_preprocessing(X_train,X_test):
  sc=StandardScaler()
  sc.fit(X_train)
  X_train_1=sc.transform(X_train)
  X_test_1=sc.transform(X_test)
  X_train_1=pd.DataFrame(X_train_1,columns=X_train.columns)
  X_test_1=pd.DataFrame(X_test_1,columns=X_test.columns)
  return(X_train_1,X_test_1)

# Creating Train and Test datasets 
X_train=pd.concat([X1,X3])
X_test=X2

y1=list(y1)
y2=list(y2)
y3=list(y3)
y_train=(y1+y3)
y_test=y2

y_train=np.array(y_train)
y_test=np.array(y_test)

X_train,X_test=Standard_Scaler_preprocessing(X_train,X_test)

# Function to obtain model performance
def perf(model,X_train,y_train,X_test,y_test):
  model.fit(X_train,y_train)
  preds=model.predict(X_test)
  print(classification_report(y_test,preds))

#Defining function for determining Model performance for DL models
def model_performance_DL(model,X_train,y_train,X_test,y_test):
  #Model is fit on the train set and then used to predict ono Test set
  y_pred = model.predict(X_test)
  predicted=[]
  predicted_prob=y_pred
  for i in range(len(y_pred)):
    predicted.append(np.argmax(y_pred[i]))
    
  
  #Different metrics are used to check the model performance
  #Weighted precision,recall,f1_score is reported for all models
  classes = np.unique(y_test)
  y_test_array = pd.get_dummies(y_test, drop_first=False).values

  accuracy = accuracy_score(y_test, predicted)
  print("Accuracy:",  round(accuracy,2))
  print("Detail:")
  print(classification_report(y_test, predicted))

  ## Plot confusion matrix
  cm = confusion_matrix(y_test, predicted)
  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
            cbar=False)
  ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
       yticklabels=classes, title="Confusion matrix")
  plt.yticks(rotation=0)

  ## Plot roc-auc curve
  plt.figure()
  for i in range(len(classes)):
    fpr, tpr, thresholds = roc_curve(y_test_array[:,i],  
                           predicted_prob[:,i])
    plt.plot(fpr, tpr, lw=3, 
              label='{0} (area={1:0.2f})'.format(classes[i], 
                              auc(fpr, tpr),figsize=(20,20))
               )

  plt.plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
  plt.xlim([-0.05,1.0])
  plt.ylim([0,1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate (Recall)')
  plt.title('Receiver operating characteristic')

  plt.legend(loc='best')
  plt.grid(True)
  
  ## Plot precision-recall curve
  plt.figure()
  for i in range(len(classes)):
    precision, recall, thresholds = precision_recall_curve(
                 y_test_array[:,i], predicted_prob[:,i])
    plt.plot(recall, precision, lw=3, 
               label='{0} (area={1:0.2f})'.format(classes[i], 
                                  auc(recall, precision))
              )
  plt.xlim([0,1.05])
  plt.ylim([0,1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve')  
  plt.legend(loc="best")
  plt.grid(True)
  plt.show()

X_train_temp=X_train.values
X_test_temp=X_test.values

#Inverse Class weights for Imbalanced classes
Tot=len(y_train)
count_arr=[0]*len(np.unique(y_train))
for i in range(len(y_train)):
  count_arr[int(y_train[i])]+=1

weight_dicts={}
for i in np.unique(y_train):
  weight_dicts[int(i)]=Tot/count_arr[int(i)]

#Inverse Class weights for Imbalanced classes
Tot=len(y_test)
count_arr=[0]*len(np.unique(y_test))
for i in range(len(y_test)):
  count_arr[int(y_test[i])]+=1

weight_dicts_test={}
for i in np.unique(y_train):
  weight_dicts_test[int(i)]=Tot/count_arr[int(i)]

#Building a ANN model
#Defining a common function to handle all the three datasets

# Building different types of DL models
def build_model(X_train,y_train,weight_dicts):
  y_train=np.asarray(y_train ,dtype=int)
  MLP_model=Sequential()
  MLP_model.add(Dense(1024,activation='relu',input_dim=len(X_train[0])))
  MLP_model.add(Dense(512,activation='relu',kernel_initializer='glorot_uniform'))
  MLP_model.add(Dense(256,activation='relu',kernel_initializer='glorot_uniform'))
  MLP_model.add(Dense(128,activation='relu',kernel_initializer='glorot_uniform'))
  MLP_model.add(Dense(64,activation='relu',kernel_initializer='glorot_uniform'))
  MLP_model.add(Dense(32,activation='relu',kernel_initializer='glorot_uniform'))
  MLP_model.add(Dense(16,activation='relu',kernel_initializer='glorot_uniform'))
  MLP_model.add(Dense(8,activation='relu',kernel_initializer='glorot_uniform'))
  MLP_model.add(Dense(5,activation='softmax'))
  MLP_model.compile(optimizer=Adam(lr=0.001),metrics=['accuracy'],loss='sparse_categorical_crossentropy',weighted_metrics=['accuracy'])
  checkpoint_filepath = '/tmp/checkpoint'
  es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
  MLP_model.fit(X_train,y_train,epochs=150,validation_split=0.2,batch_size=64,callbacks=[es],class_weight=weight_dicts)
  MLP_model.load_weights(checkpoint_filepath)
  return(MLP_model)


def build_model_LSTM_CNN(X_train,y_train,weight_dicts):
  y_train=np.asarray(y_train ,dtype=int)
  y_train=to_categorical(y_train,num_classes=5)
  X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],-1)
  model=Sequential()
  model.add(Conv1D(filters=128,kernel_size=5,padding='same',input_shape=(X_train.shape[1],X_train.shape[2])))
  model.add(SpatialDropout1D(0.1))
  model.add(LSTM(256,return_sequences=True))
  model.add(LSTM(256,return_sequences=True))
  model.add(Flatten())
  model.add(Dense(64,activation='relu'))
  model.add(Dense(32,activation='relu'))
  model.add(Dense(16,activation='relu'))
  model.add(Dense(8,activation='relu'))
  model.add(Dense(5,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'],weighted_metrics=['accuracy'])
  checkpoint_filepath = '/tmp/checkpoint'
  es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
  model.fit(X_train,y_train,epochs=100,validation_split=0.2,batch_size=64,callbacks=[es],class_weight=weight_dicts)
  model.load_weights(checkpoint_filepath)
  return(model)

def build_model_LSTM(X_train,y_train,weight_dicts):
  y_train=np.asarray(y_train ,dtype=int)
  y_train=to_categorical(y_train,num_classes=5)
  X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],-1)
  model=Sequential()
  model.add(LSTM(256,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])))
  model.add(LSTM(256,return_sequences=True))
  model.add(LSTM(256,return_sequences=True))
  model.add(Flatten())
  model.add(Dense(64,activation='relu'))
  model.add(Dense(32,activation='relu'))
  model.add(Dense(16,activation='relu'))
  model.add(Dense(8,activation='relu'))
  model.add(Dense(5,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'],weighted_metrics=['accuracy'])
  checkpoint_filepath = '/tmp/checkpoint'
  es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
  model.fit(X_train,y_train,epochs=150,validation_split=0.2,batch_size=64,callbacks=[es],class_weight=weight_dicts)
  model.load_weights(checkpoint_filepath)
  return(model)

#For finding out the performance of DL models use this cell
#So as to execute some other other than LSTM
# Use the corresponding build function 
MLP_model_train=build_model(X_train_temp,y_train,weight_dicts)
model_performance_DL(MLP_model_train,X_train_temp,y_train,X_test_temp,y_test)
weights=MLP_model_train.layers[-1].get_weights()


# Transfer Learning Model
model=Sequential()
X_train1=X_train_temp.reshape(X_train_temp.shape[0],X_train_temp.shape[1],-1)
model.add(LSTM(256,return_sequences=True,input_shape=(X_train1.shape[1],X_train1.shape[2])))
model.add(LSTM(256,return_sequences=True))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(5,activation='softmax'))


model.layers[-1].set_weights(weights)
model.layers[-1].trainable=False


X_t1,X_t2,y_t1,y_t2=train_test_split(X_test,y_test,test_size=0.1,random_state=2)

X_t=X_t1.values
X_t=X_t.reshape(X_t.shape[0],X_t.shape[1],-1)

#Target domain DL model
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'],weighted_metrics=['accuracy'])
checkpoint_filepath = '/tmp/checkpoint'
es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
model.fit(X_t,y_t1,epochs=100,validation_split=0.2,batch_size=64,callbacks=[es])
model.load_weights(checkpoint_filepath)

# Performance of Target domain model
X_t3=X_t2.values
X_t3=X_t3.reshape(X_t3.shape[0],X_t3.shape[1],-1)
model_performance_DL(model,X_t,y_t1,X_t3,y_t2)
