# CS9542 Final Project
# This script builds + trains all models. Models have different network architectures specified in the report.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
import numpy as np
import os

## LOAD CLAHE-PROCESSED-2 DATA ###################################################################
train_ds_clahe_param2 = tf.keras.preprocessing.image_dataset_from_directory(
    directory           = "Dataset/Train/Preprocessed/Clahe2",
    seed                = 1337,
    label_mode          = "categorical",
    image_size          = (299,299),
    batch_size          = 32
)
#############################################################################################

## LOAD CLAHE-PROCESSED-3 DATA ###################################################################
train_ds_clahe_param3 = tf.keras.preprocessing.image_dataset_from_directory(
    directory           = "Dataset/Train/Preprocessed/Clahe3",
    seed                = 1337,
    label_mode          = "categorical",
    image_size          = (299,299),
    batch_size          = 32
)
#############################################################################################


#############################################################################################
def model1__2_64_256(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(256, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model2__2_64_512(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(512, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model3__2_64_1028(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(1028, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model4__2_128_256(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(128, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(256, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model5__2_128_512(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(128, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(512, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model6__2_128_1028(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(128, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(1028, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model7__3_64_256(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(256, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model8__3_64_512(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(512, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model

#############################################################################################
def model9__3_64_1028(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(1028, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model10__3_128_256(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(128, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(256, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model11__3_128_512(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(128, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(512, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################
def model12__3_128_1028(input_shape):
    model = tf.keras.Sequential()
    #First convolution
    model.add( layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape) )
    model.add( layers.MaxPooling2D(2,2) )
    #Second convolution
    model.add( layers.Conv2D(32, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Third convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fourth convolution
    model.add( layers.Conv2D(64, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Fifth convolution
#TUNING:
#-->[try: 64, 128 params (where it currently says "64")]
    model.add( layers.Conv2D(128, (3,3), activation='relu') )
    model.add( layers.MaxPooling2D(2,2) )
    #Flatten -> Dense -> 512 neurons -> softmax (for normal vs COVID vs Non-Covid Pneumonia)
#TUNING:
#-->[try: 256, 512, 1028 neurons (where it currently says "512")]
    model.add( layers.Flatten() )
    model.add( layers.Dense(1028, activation='relu') )
    model.add( layers.Dense(3, activation='softmax') ) #multi-classification (3 classes)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),loss='categorical_crossentropy',metrics = ['accuracy'])
    return model
#############################################################################################


input_shape = (299,299,3)
epochs=10
model1 = model1__2_64_256(input_shape)
model2 = model2__2_64_512(input_shape)
model3 = model3__2_64_1028(input_shape)
model4 = model4__2_128_256(input_shape)
model5 = model5__2_128_512(input_shape)
model6 = model6__2_128_1028(input_shape)
model7 = model7__3_64_256(input_shape)
model8 = model8__3_64_512(input_shape)
model9  = model9__3_64_1028(input_shape)
model10 = model10__3_128_256(input_shape)
model11 = model11__3_128_512(input_shape)
model12 = model12__3_128_1028(input_shape)


print()
print("Training models...")
print()
#DS_param2:
model1.fit( train_ds_clahe_param2, epochs=epochs )
model2.fit( train_ds_clahe_param2, epochs=epochs )
model3.fit( train_ds_clahe_param2, epochs=epochs )
model4.fit( train_ds_clahe_param2, epochs=epochs )
model5.fit( train_ds_clahe_param2, epochs=epochs )
model6.fit( train_ds_clahe_param2, epochs=epochs )
#DS_param3:
model7.fit( train_ds_clahe_param3, epochs=epochs )
model8.fit( train_ds_clahe_param3, epochs=epochs )
model9.fit(  train_ds_clahe_param3, epochs=epochs )
model10.fit( train_ds_clahe_param3, epochs=epochs )
model11.fit( train_ds_clahe_param3, epochs=epochs )
model12.fit( train_ds_clahe_param3, epochs=epochs )

print()
print("Finished training.")
print()

# Save model after training is complete
print()
print("Saving models...")
print()
# clahe_model.save("clahe_model")
model1.save( "model1" )
model2.save( "model2" )
model3.save( "model3" )
model4.save( "model4" )
model5.save( "model5" )
model6.save( "model6" )
#DS_param3:
model7.save( "model7" )
model8.save( "model8" )
model9.save(  "model9" )
model10.save( "model10" )
model11.save( "model11" )
model12.save( "model12" )

print()
print()
print()
print("SCRIPT FINISHED...")
print()
print()
