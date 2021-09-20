import pandas as pd
import numpy as np 

from keras.models import Sequential
from keras.layers import Dense,Dropout,ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers import Embedding,Conv1D,LSTM,Input,TimeDistributed,SpatialDropout1D,Flatten,Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint

#Importing models from sklearn library
#Trying various inbuilt models
#SVM,Naive bayes,KNN,Random Forest, Decision Trees
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


#Fpr Building a ANN model
#Importing necssary layers from keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


#Importing the metrics to compare the model performance
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import matthews_corrcoef,confusion_matrix


# Loading data
# Modify path to each datasets accordingly
data_us=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Thermal sensation prediction(Murata)/Medium_office.csv')
data_ash=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Thermal sensation prediction(Murata)/Ashrae_clean.csv')

# CLASSIFICATION MODELS

data_us_C=data_us
data_us_B=data_us.drop(['Outdoor monthly air temperature (¡C)'],axis=1)
data_us_A=data_us_B.drop(['Age','Sex'],axis=1)

data_B=data_ash.drop(['Outdoor monthly air temperature (¡C)'],axis=1)
data_A=data_B.drop(['Age','Sex'],axis=1)

def fun1(x):
  return(np.round(x,2))

cols=['Air velocity (m/s)','Air temperature (¡C)','Radiant temperature (¡C)','Relative humidity (%)']  
for col in cols:
  data_us_A[col]=data_us_A[col].apply(lambda x:fun1(x))
  data_us_B[col]=data_us_B[col].apply(lambda x:fun1(x))
  data_us_C[col]=data_us_C[col].apply(lambda x:fun1(x))


#Defining the models
#Using Class weights as Balanced(Hueristic approach to handle imbalanced classes)
SVM_lin_model=SVC(class_weight='balanced',kernel='linear')
SVM_poly_model=SVC(class_weight='balanced',kernel='poly')
SVM_radial_model=SVC(class_weight='balanced',kernel='rbf')
NB_model=GaussianNB()
#Instead of giving equal weights to all neighbours, Here the weights are inversely 
#proptional to the distance of the point from the desired point
KNN_model=KNeighborsClassifier(n_neighbors=10,weights='distance')
RF_model=RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample')
DT_model=DecisionTreeClassifier(max_depth=5,class_weight='balanced')
Adaboost_model=AdaBoostClassifier(n_estimators=200,random_state=1)


#Defining a common function to handle all the three datasets
#I tried adding Dropouts, Performance decreased hence I have removed them.
#Tried different intializers also, not much difference 
def build_model(X_train,y_train):
  y_train=np.asarray(y_train ,dtype=int)
  MLP_model=Sequential()
  sums=[0,0,0,0,0]
  for i in range(len(y_train)):
    sums[y_train[i]]+=1
  MLP_model.add(Dense(1024,activation='relu',input_shape=(len(list(X_train.columns)),)))
  MLP_model.add(Dense(512,activation='relu',kernel_initializer='he_uniform'))
  MLP_model.add(Dense(256,activation='relu',kernel_initializer='he_uniform'))
  MLP_model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
  MLP_model.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
  MLP_model.add(Dense(32,activation='relu',kernel_initializer='he_uniform'))
  MLP_model.add(Dense(16,activation='relu',kernel_initializer='he_uniform'))
  MLP_model.add(Dense(8,activation='relu',kernel_initializer='he_uniform'))
  MLP_model.add(Dense(5,activation='softmax'))
  MLP_model.compile(optimizer=Adam(lr=0.001),metrics=['accuracy'],loss='sparse_categorical_crossentropy')
  #Early stopping to prevent overfitting 
  #es=EarlyStopping(monitor='val_loss',patience=2,restore_best_weights=True)
  MLP_model.fit(X_train,y_train,epochs=20,validation_split=0.2,batch_size=32,class_weight={0:1/sums[0],1:1/sums[1],2:1/sums[2],3:1/sums[3],4:1/sums[4]})
  return(MLP_model)

#This functions trains the model and performs prediction and anlaysis of model  performance
def model_performance(model,X_test,y_test,model_name):
  y_pred=model.predict(X_test)
  acc=accuracy_score(y_test,y_pred)
  cm=confusion_matrix(y_test,y_pred)
  recall = np.diag(cm) / np.sum(cm, axis = 1)
  precision = np.diag(cm) / np.sum(cm, axis = 0)
  precs=np.mean(recall)
  recall=np.mean(precision)
  f1=(2*precs*recall)/(precs+recall)
  coeff=matthews_corrcoef(y_test,y_pred)
  print('Accuracy of {} is: {}'.format(model_name,acc))
  print('F1_Score of {} is: {}'.format(model_name,f1))
  print('Precsion of {} is: {}'.format(model_name,precs))
  print('Recall of {} is: {}'.format(model_name,recall))
  print('Mathew_coeff of {} is: {} '.format(model_name,coeff))
  print('Confusion matrix of {} is:'.format(model_name))
  print(cm)

#Importing the metrics to compare the model performance
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import matthews_corrcoef,confusion_matrix
#This functions trains the model and performs prediction and anlaysis of model  performance
def model_performance_MLP(model,X_test,y_test,model_name):
  y_pred=model.predict(X_test)
  y_preds=[]
  for i in range(len(y_pred)):
    y_preds.append(np.argmax(y_pred[i]))

  y_pred=y_preds
  acc=accuracy_score(y_test,y_pred)
  cm=confusion_matrix(y_test,y_pred)
  recall = np.diag(cm) / np.sum(cm, axis = 1)
  precision = np.diag(cm) / np.sum(cm, axis = 0)
  precs=np.mean(recall)
  recall=np.mean(precision)
  f1=(2*precs*recall)/(precs+recall)
  coeff=matthews_corrcoef(y_test,y_pred)
  print('Accuracy of {} is: {}'.format(model_name,acc))
  print('F1_Score of {} is: {}'.format(model_name,f1))
  print('Precsion of {} is: {}'.format(model_name,precs))
  print('Recall of {} is: {}'.format(model_name,recall))
  print('Mathew_coeff of {} is: {} '.format(model_name,coeff))
  print('Confusion matrix of {} is:'.format(model_name))
  print(cm)



models=[SVM_lin_model,SVM_poly_model,SVM_radial_model,NB_model,KNN_model,RF_model,DT_model,Adaboost_model,MLP_model]

X_train=data_C.drop(['Thermal sensation'],axis=1)
y_train=data_C['Thermal sensation']
X_test=data_us_C.drop(['Thermal sensation'],axis=1)
y_test=data_us_C['Thermal sensation']
model=RF_model
model.fit(X_train,y_train)
model_performance(model,X_test,y_test,str(model))



def build_model_LSTM(X_train,y_train):
  y_train=np.asarray(y_train ,dtype=int)
  sums=[0,0,0,0,0]
  for i in range(len(y_train)):
    sums[y_train[i]]+=1
  y_train=to_categorical(y_train,num_classes=5)
  model=Sequential()
  model.add(Embedding(101,256,input_length=len(X_train.columns),))
  model.add(LSTM(256,return_sequences=True))
  model.add(LSTM(256,return_sequences=True))
  model.add(Flatten())
  model.add(Dense(64,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(16,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(5,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001,beta_1=0.99,beta_2=0.999),metrics=['accuracy'],weighted_metrics=['accuracy'])
  es=EarlyStopping(monitor='val_loss',restore_best_weights=True)
  model.fit(X_train,y_train,epochs=20,validation_split=0.2,batch_size=64,class_weight={0:1/sums[0],1:1/sums[1],2:1/sums[2],3:1/sums[3],4:1/sums[4]})
  model.summary()
  return(model)

X_train=data_A.drop(['Thermal sensation'],axis=1)
X_train=X_train.drop(['Unnamed: 0'],axis=1)
y_train=data_A['Thermal sensation']
X_test=data_us_A.drop(['Thermal sensation'],axis=1)
X_test=X_test.drop(['Unnamed: 0'],axis=1)
y_test=data_us_A['Thermal sensation']
model=build_model_LSTM(X_train,y_train)
model_performance_MLP(model,X_test,y_test,str(model))
