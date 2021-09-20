import pandas as pd
import numpy as np 

#Importing models from sklearn library
#Trying various inbuilt models
#SVM,Naive bayes,KNN,Random Forest, Decision Trees

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn import *

#For plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Reading ASHRAE AND SCALES DATA
# Modify data path accordingly

data_us=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Thermal sensation prediction(Murata)/Medium_office.csv')
data_C=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Thermal sensation prediction(Murata)/Ashrae_clean.csv')

data_C.drop(['Unnamed: 0'],axis=1,inplace=True)
data_C=data_C.drop_duplicates(['Age',	'Sex','Air velocity (m/s)',	'Air temperature (¡C)',	'Radiant temperature (¡C)',	'Relative humidity (%)',	'Clo',	'Met',	'Outdoor monthly air temperature (¡C)'])

data_us.drop(['Unnamed: 0'],axis=1,inplace=True)
data_us=data_us.drop_duplicates(['Age',	'Sex','Air velocity (m/s)',	'Air temperature (¡C)',	'Radiant temperature (¡C)',	'Relative humidity (%)',	'Clo',	'Met',	'Outdoor monthly air temperature (¡C)'])

data_us_C=data_us

def fun1(x):
  return(np.round(x,2))

cols=['Air velocity (m/s)','Air temperature (¡C)','Radiant temperature (¡C)','Relative humidity (%)']  
for col in cols:
  data_us_C[col]=data_us_C[col].apply(lambda x:fun1(x))







#Plotting the Number of sample in each class in train data
fig, ax = plt.subplots()
fig.suptitle("Thermal Sensation", fontsize=12)
data_C["Thermal sensation"].reset_index().groupby("Thermal sensation").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()

#Plotting the Number of sample in each class in Test data
fig, ax = plt.subplots()
fig.suptitle("Thermal Sensation", fontsize=12)
data_us_C["Thermal sensation"].reset_index().groupby("Thermal sensation").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()

#Creating the Train and Test dataframes
data_C=data_C[data_C>0]
data_C=data_C.dropna()
X_train=data_C.drop(['Thermal sensation'],axis=1)
y_train=data_C['Thermal sensation'].values


data_us_C=data_us_C[data_us_C>0]
data_us_C=data_us_C.dropna()
X_test=data_us_C.drop(['Thermal sensation'],axis=1)
y_test=data_us_C['Thermal sensation'].values





#Feature selection
#Set of all features which is statistically significant in train set
y = data_C["Thermal sensation"]
X_names = X_train.columns
p_value_limit = 0.95
dtf_features = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X_train, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[True,False])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()

print('Important Columns are :',X_names)

#The feature playing a major role in identifying each category
#Statistical significance limit is set at alpha =0.05
for cat in np.unique(y):
   print("# {}:".format(cat))
   print("  . selected features:",
         len(dtf_features[dtf_features["y"]==cat]))
   print("  . top features:", ",".join(
dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
   print(" ")

#Selecting only the cols which are statistically signoificant
X_train=X_train[X_names]
X_test=X_test[X_names]





#Feature selection
#Set of all features which is statistically significant in train set
y = data_us_C["Thermal sensation"]
X_names_test = X_test.columns
p_value_limit = 0.95
dtf_features_test = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X_test, y==cat)
    dtf_features_test = dtf_features_test.append(pd.DataFrame(
                   {"feature":X_names_test, "score":1-p, "y":cat}))
    dtf_features_test = dtf_features_test.sort_values(["y","score"], 
                    ascending=[True,False])
    dtf_features_test = dtf_features_test[dtf_features_test["score"]>p_value_limit]
X_names_test = dtf_features_test["feature"].unique().tolist()


#The feature playing a major role in identifying each category
#Statistical significance limit is set at alpha =0.05
for cat in np.unique(y):
   print("# {}:".format(cat))
   print("  . selected features:",
         len(dtf_features_test[dtf_features_test["y"]==cat]))
   print("  . top features:", ",".join(
dtf_features_test[dtf_features_test["y"]==cat]["feature"].values[:10]))
   print(" ")

#Defining the models
#Using Class weights as Balanced(Hueristic approach to handle imbalanced classes)
SVM_lin_model=SVC(class_weight='balanced',kernel='linear')
SVM_poly_model=SVC(class_weight='balanced',kernel='poly')
SVM_radial_model=SVC(class_weight='balanced',kernel='rbf')
NB_model=GaussianNB()
#Instead of giving equal weights to all neighbours, Here the weights are inversely 
#proptional to the distance of the point from the desired point
KNN_model=KNeighborsClassifier(n_neighbors=10,weights='distance')
RF_model=RandomForestClassifier(n_estimators=200,class_weight='balanced_subsample',max_depth=10)
DT_model=DecisionTreeClassifier(max_depth=10,class_weight='balanced')
Adaboost_model=AdaBoostClassifier(n_estimators=200,random_state=1)

#Defining function

def model_performance(model,X_train,y_train,X_test,y_test):
  #Model is fit on the train set and then used to predict ono Test set
  model.fit(X_train,y_train)
  predicted = model.predict(X_test)
  predicted_prob = model.predict_proba(X_test)
  
  #Different metrics are used to check the model performance
  #Weighted precision,recall,f1_score is reported for all models
  classes = np.unique(y_test)
  y_test_array = pd.get_dummies(y_test, drop_first=False).values

  accuracy = metrics.accuracy_score(y_test, predicted)
  auc = metrics.roc_auc_score(y_test, predicted_prob, 
                            multi_class="ovr")
  print("Accuracy:",  round(accuracy,2))
  print("Auc:", round(auc,2))
  print("Detail:")
  print(metrics.classification_report(y_test, predicted))

  ## Plot confusion matrix
  cm = metrics.confusion_matrix(y_test, predicted)
  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
            cbar=False)
  ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
       yticklabels=classes, title="Confusion matrix")
  plt.yticks(rotation=0)

  ## Plot roc-auc curve
  plt.figure()
  for i in range(len(classes)):
    fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],  
                           predicted_prob[:,i])
    plt.plot(fpr, tpr, lw=3, 
              label='{0} (area={1:0.2f})'.format(classes[i], 
                              metrics.auc(fpr, tpr),figsize=(20,20))
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
    precision, recall, thresholds = metrics.precision_recall_curve(
                 y_test_array[:,i], predicted_prob[:,i])
    plt.plot(recall, precision, lw=3, 
               label='{0} (area={1:0.2f})'.format(classes[i], 
                                  metrics.auc(recall, precision))
              )
  plt.xlim([0,1.05])
  plt.ylim([0,1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve')  
  plt.legend(loc="best")
  plt.grid(True)
  plt.show()

#Change the model name accordingle to get the performance of desired model
model_performance(Adaboost_model,X_train,y_train,X_test,y_test)

#Plotting Prercision recall curve for each classifier for each category separately
classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values

models=[NB_model,KNN_model,RF_model,DT_model,Adaboost_model]
model_names=['NB_model','KNN_model','RF_model','DT_model','Adaboost_model']
for i in range(len(classes)):
  plt.figure()
  plt.xlim([0,1.05])
  plt.ylim([0,1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve') 
  
  plt.grid(True)
  count=0
  for model in models:
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)
    precision, recall, thresholds = metrics.precision_recall_curve(
                 y_test_array[:,i], predicted_prob[:,i])
    plt.plot(recall, precision, lw=3, 
               label='{} of {}'.format(classes[i],model_names[count]))
    count+=1           
  plt.legend(loc="best")  
  plt.show()



#Building a ANN model
#Importing necssary layers from keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers import Embedding,Conv1D,LSTM,Input,TimeDistributed,SpatialDropout1D,Flatten,Dropout
from keras.models import Model

#Defining a common function to handle all the three datasets
#Tried different intializers also, not much difference 
def build_model(X_train,y_train):
  y_train=np.asarray(y_train ,dtype=int)
  MLP_model=Sequential()
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
  MLP_model.fit(X_train,y_train,epochs=20,validation_split=0.2,batch_size=64)
  return(MLP_model)

def build_model_LSTM_CNN(X_train,y_train):
  y_train=np.asarray(y_train ,dtype=int)
  y_train=to_categorical(y_train,num_classes=5)
  model=Sequential()
  model.add(Embedding(101,256,input_length=len(X_train.columns),))
  model.add(Conv1D(filters=128,kernel_size=5,padding='same'))
  model.add(SpatialDropout1D(0.1))
  model.add(LSTM(256,return_sequences=True))
  model.add(LSTM(256,return_sequences=True))
  model.add(Flatten())
  model.add(Dense(64,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(16,activation='relu'))
  model.add(Dropout(0.1))
  model.add(Dense(5,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
  #es=EarlyStopping(monitor='val_loss',restore_best_weights=True)
  model.fit(X_train,y_train,epochs=20,validation_split=0.2,batch_size=64)
  model.summary()
  return(model)

def build_model_LSTM(X_train,y_train):
  y_train=np.asarray(y_train ,dtype=int)
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
  model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
  #es=EarlyStopping(monitor='val_loss',restore_best_weights=True)
  model.fit(X_train,y_train,epochs=20,validation_split=0.2,batch_size=64)
  model.summary()
  return(model)

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

  accuracy = metrics.accuracy_score(y_test, predicted)
  print("Accuracy:",  round(accuracy,2))
  print("Auc:", round(auc,2))
  print("Detail:")
  print(metrics.classification_report(y_test, predicted))

  ## Plot confusion matrix
  cm = metrics.confusion_matrix(y_test, predicted)
  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
            cbar=False)
  ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
       yticklabels=classes, title="Confusion matrix")
  plt.yticks(rotation=0)

  ## Plot roc-auc curve
  plt.figure()
  for i in range(len(classes)):
    fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],  
                           predicted_prob[:,i])
    plt.plot(fpr, tpr, lw=3, 
              label='{0} (area={1:0.2f})'.format(classes[i], 
                              metrics.auc(fpr, tpr),figsize=(20,20))
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
    precision, recall, thresholds = metrics.precision_recall_curve(
                 y_test_array[:,i], predicted_prob[:,i])
    plt.plot(recall, precision, lw=3, 
               label='{0} (area={1:0.2f})'.format(classes[i], 
                                  metrics.auc(recall, precision))
              )
  plt.xlim([0,1.05])
  plt.ylim([0,1.05])
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall curve')  
  plt.legend(loc="best")
  plt.grid(True)
  plt.show()

#For finding out the performance of DL models use this cell
#So as to execute some other other than LSTM
# Use the corresponding build function 
LSTM_model=build_model_LSTM(X_train,y_train)
model_performance_DL(LSTM_model,X_train,y_train,X_test,y_test)









