import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#LOAD TEST DATA
dataset_clahe_2 = tf.keras.preprocessing.image_dataset_from_directory(
    directory           = "Dataset/Test/Preprocessed/Clahe2",
    seed                = 1337,
    label_mode          = "categorical",
    image_size          = (299,299),
    batch_size          = 32
)

#Load models
print("\n\nLoading models...\n")
#clahe=2 models:
model1 = keras.models.load_model("model1")
model2 = keras.models.load_model("model2")
model3 = keras.models.load_model("model3")
model4 = keras.models.load_model("model4")
model5 = keras.models.load_model("model5")
model6 = keras.models.load_model("model6")
print("\n\nModels loaded...\n")
print("\nRunning Predictions...\n")
#run inference - clahe=2 models 1-6:

predictions = np.array([])
labels =  np.array([])
for x, y in dataset_clahe_2:
  predictions = np.concatenate([predictions, np.argmax(model1.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
predVSactual = np.zeros((2002,2))
predVSactual[:,0] = labels #actual
predVSactual[:,1] = predictions
dVSactual_df = pd.DataFrame(data=predVSactual,columns=['actual','predicted'])
dVSactual_df.to_csv('C:\\Users\\Admin\\Documents\\CS9542CovidProject\\results1.csv')

predictions = np.array([])
labels =  np.array([])
for x, y in dataset_clahe_2:
  predictions = np.concatenate([predictions, np.argmax(model2.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
predVSactual = np.zeros((2002,2))
predVSactual[:,0] = labels #actual
predVSactual[:,1] = predictions
dVSactual_df = pd.DataFrame(data=predVSactual,columns=['actual','predicted'])
dVSactual_df.to_csv('C:\\Users\\Admin\\Documents\\CS9542CovidProject\\results2.csv')

predictions = np.array([])
labels =  np.array([])
for x, y in dataset_clahe_2:
  predictions = np.concatenate([predictions, np.argmax(model3.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
predVSactual = np.zeros((2002,2))
predVSactual[:,0] = labels #actual
predVSactual[:,1] = predictions
dVSactual_df = pd.DataFrame(data=predVSactual,columns=['actual','predicted'])
dVSactual_df.to_csv('C:\\Users\\Admin\\Documents\\CS9542CovidProject\\results3.csv')


predictions = np.array([])
labels =  np.array([])
for x, y in dataset_clahe_2:
  predictions = np.concatenate([predictions, np.argmax(model4.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
predVSactual = np.zeros((2002,2))
predVSactual[:,0] = labels #actual
predVSactual[:,1] = predictions
dVSactual_df = pd.DataFrame(data=predVSactual,columns=['actual','predicted'])
dVSactual_df.to_csv('C:\\Users\\Admin\\Documents\\CS9542CovidProject\\results4.csv')


predictions = np.array([])
labels =  np.array([])
for x, y in dataset_clahe_2:
  predictions = np.concatenate([predictions, np.argmax(model5.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
predVSactual = np.zeros((2002,2))
predVSactual[:,0] = labels #actual
predVSactual[:,1] = predictions
dVSactual_df = pd.DataFrame(data=predVSactual,columns=['actual','predicted'])
dVSactual_df.to_csv('C:\\Users\\Admin\\Documents\\CS9542CovidProject\\results5.csv')


predictions = np.array([])
labels =  np.array([])
for x, y in dataset_clahe_2:
  predictions = np.concatenate([predictions, np.argmax(model6.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
predVSactual = np.zeros((2002,2))
predVSactual[:,0] = labels #actual
predVSactual[:,1] = predictions
dVSactual_df = pd.DataFrame(data=predVSactual,columns=['actual','predicted'])
dVSactual_df.to_csv('C:\\Users\\Admin\\Documents\\CS9542CovidProject\\results6.csv')

print("\nPredictions Finished + Expoted.")

print("\n\nSCRIPT FINISHED.\n\n")
