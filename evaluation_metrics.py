import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# load test predictions
df1 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results1.csv")
df1.drop(columns = 'Unnamed: 0',inplace=True)
df2 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results2.csv")
df2.drop(columns = 'Unnamed: 0',inplace=True)
df3 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results3.csv")
df3.drop(columns = 'Unnamed: 0',inplace=True)
df4 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results4.csv")
df4.drop(columns = 'Unnamed: 0',inplace=True)
df5 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results5.csv")
df5.drop(columns = 'Unnamed: 0',inplace=True)
df6 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results6.csv")
df6.drop(columns = 'Unnamed: 0',inplace=True)
df7 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results7.csv")
df7.drop(columns = 'Unnamed: 0',inplace=True)
df8 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results8.csv")
df8.drop(columns = 'Unnamed: 0',inplace=True)
df9 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results9.csv")
df9.drop(columns = 'Unnamed: 0',inplace=True)
df10 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results10.csv")
df10.drop(columns = 'Unnamed: 0',inplace=True)
df11 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results11.csv")
df11.drop(columns = 'Unnamed: 0',inplace=True)
df12 = pd.read_csv("/Users/ryandiv/miniforge3/envs/tensorflowVE/CS9542ScriptsOnly/results12.csv")
df12.drop(columns = 'Unnamed: 0',inplace=True)

# computing accuracy
def accuracy(df):
    correctCounter = 0
    for index in range((df.shape[0])):
        if df.iloc[index][0] == df.iloc[index][1]:
            correctCounter += 1
    return (round((correctCounter / df.shape[0]),2))*100

modelAcc1 = accuracy(df1)
modelAcc2 = accuracy(df2)
modelAcc3 = accuracy(df3)
modelAcc4 = accuracy(df4)
modelAcc5 = accuracy(df5)
modelAcc6 = accuracy(df6)

# generate remaining evaluation metrics
yActual     = df1['actual'].to_numpy()
yPredicted  = df1['predicted'].to_numpy()
cf1 = metrics.confusion_matrix(yActual, yPredicted)
cr1 = metrics.classification_report(yActual, yPredicted, digits=3)
print(cr1)
print(cf1)

yActual     = df2['actual'].to_numpy()
yPredicted  = df2['predicted'].to_numpy()
cf2 = metrics.confusion_matrix(yActual, yPredicted)
cr2 = metrics.classification_report(yActual, yPredicted, digits=3)
print(cr2)
print(cf2)

yActual     = df3['actual'].to_numpy()
yPredicted  = df3['predicted'].to_numpy()
cf3 = metrics.confusion_matrix(yActual, yPredicted)
cr3 = metrics.classification_report(yActual, yPredicted, digits=3)
print(cr3)
print(cf3)

yActual     = df4['actual'].to_numpy()
yPredicted  = df4['predicted'].to_numpy()
cf4 = metrics.confusion_matrix(yActual, yPredicted)
cr4 = metrics.classification_report(yActual, yPredicted, digits=3)
print(cr4)
print(cf4)

yActual     = df5['actual'].to_numpy()
yPredicted  = df5['predicted'].to_numpy()
cf5 = metrics.confusion_matrix(yActual, yPredicted)
cr5 = metrics.classification_report(yActual, yPredicted, digits=3)
print(cr5)
print(cf5)

yActual     = df6['actual'].to_numpy()
yPredicted  = df6['predicted'].to_numpy()
cf6 = metrics.confusion_matrix(yActual, yPredicted)
cr6 = metrics.classification_report(yActual, yPredicted, digits=3)
print(cr6)
print(cf6)
