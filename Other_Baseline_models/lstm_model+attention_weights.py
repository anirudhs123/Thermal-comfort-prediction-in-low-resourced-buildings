
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

from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

from keras.models import Sequential
from keras.layers import Dense,Dropout,ReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils import to_categorical
from keras.layers import Embedding,Conv1D,LSTM,Input,TimeDistributed,SpatialDropout1D,Flatten,Dropout
from keras.models import Model


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

#Reading ASHRAE AND SCALES DATA
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

#Creating the Train and Test dataframes
data_C=data_C[data_C>0]
data_C=data_C.dropna()
X_train=data_C.drop(['Thermal sensation'],axis=1)
y_train=data_C['Thermal sensation'].values


data_us_C=data_us_C[data_us_C>0]
data_us_C=data_us_C.dropna()
X_test=data_us_C.drop(['Thermal sensation'],axis=1)
y_test=data_us_C['Thermal sensation'].values


# Make an instance of the Model
pca = PCA(.99)
pca.fit(X_train)
X_train_1 = pca.transform(X_train)
X_test_1 = pca.transform(X_test)




sc=MinMaxScaler()
sc.fit(X_train_1)
X_train_2=sc.transform(X_train_1)
X_test_2=sc.transform(X_test_1)

## attention layer
def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

## input
x_in = layers.Input(shape=(X_train.shape[1],))
## embedding
x = layers.Embedding(input_dim=101,  
                     output_dim=256, 
                     input_length=X_train.shape[0])(x_in)

## apply attention
x = attention_layer(x, neurons=X_train.shape[1])
## 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, 
                         return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(5, activation='softmax')(x)
## compile
model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()



## train
checkpoint_filepath = '/tmp/checkpoint'
es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_loss',save_best_only=True,mode='min',save_weights_only=True)
training = model.fit(x=X_train, y=y_train, batch_size=64, 
                     epochs=20,verbose=1, validation_split=0.2,callbacks=[es])
model.load_weights(checkpoint_filepath)

#Plotting variation Training and Validation Loss vs Epochs
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
ax11.set_ylabel("Score", color='steelblue')
ax11.legend()
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics:
     ax22.plot(training.history['val_'+metric], label=metric)
ax22.set_ylabel("Score", color="steelblue")
plt.show()

#Defining function for determining Model performance for DL models
from sklearn.metrics import *
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
  #print("Auc:", round(auc,2))
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

model_performance_DL(model,X_train,y_train,X_test,y_test)


col_dict={}
for col in X_train.columns:
  col_dict[col]=[]

for i in range(len(X_test)):
  X_instance=X_test.iloc[i].values
  X_instance=X_instance.reshape(1,X_instance.shape[0],1)

  #attention weights
  layer = [layer for layer in model.layers if "attention" in 
           layer.name][0]
  func = K.function([model.input], [layer.output])
  weights = func(X_instance)[0]
  weights = np.mean(weights, axis=2).flatten()


  #rescale weights, remove null vector, map word-weight
  weights = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(np.array(weights).reshape(-1,1)).reshape(-1)
  weights = [weights[n] for n,idx in enumerate(X_instance[0]) if idx 
             != 0]


  dict_col_weight = {word:weights[n] for n,word in 
                    enumerate(X_train.columns)}
  for col in dict_col_weight.keys():
    col_dict[col].append(dict_col_weight[col])

for col in col_dict.keys():
  col_dict[col]=np.mean(col_dict[col])

# Average weigt of cols barplot
if len(col_dict) > 0:
   dtf = pd.DataFrame.from_dict(col_dict, orient='index', 
                                columns=["score"])
   dtf.sort_values(by="score", 
           ascending=True).plot(kind="barh", 
           legend=False).grid(axis='x')
   plt.show()
