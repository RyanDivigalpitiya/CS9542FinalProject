import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tqdm import tqdm
import numpy as np
import cv2
import os


## create path variables for the dataset images

# binary_healthy_path = "CS9542FinalProject/Dataset/HealthyVsPneumonia/Healthy"
# binary_pneumonia_path = "CS9542FinalProject/Dataset/HealthyVsPneumonia/Pneumonia"

## ORIGINAL TRAIN DATASET LOCATION ##################################################################################################
multiclassifier_train_covid_path_original      = "Dataset/Train/Unprocessed/Covid"
multiclassifier_train_healthy_path_original    = "Dataset/Train/Unprocessed/Healthy"
multiclassifier_train_noncovid_path_original   = "Dataset/Train/Unprocessed/Non-Covid"
## ORIGINAL TEST DATASET LOCATION ##################################################################################################
multiclassifier_test_covid_path_original      = "Dataset/Test/Unprocessed/Covid"
multiclassifier_test_healthy_path_original    = "Dataset/Test/Unprocessed/Healthy"
multiclassifier_test_noncovid_path_original   = "Dataset/Test/Unprocessed/Non-Covid"
###############################################################################################################################
## PRE-PROCESSED-Hist DATASET EXPORT LOCATION ######################################################################################
# multiclassifier_covid_path_hist_preprocessed      = "Dataset/Preprocessed/Hist/HealthyVsCovidVsNonCovid/Covid"
# multiclassifier_healthy_path_hist_preprocessed    = "Dataset/Preprocessed/Hist/HealthyVsCovidVsNonCovid/Healthy"
# multiclassifier_noncovid_path_hist_preprocessed   = "Dataset/Preprocessed/Hist/HealthyVsCovidVsNonCovid/Non-Covid"
###############################################################################################################################
## PRE-PROCESSED-Clahe >> 2.0 << TRAIN DATASET EXPORT LOCATION ######################################################################################
multiclassifier_covid_path_train_Clahe2_preprocessed      = "Dataset/Train/Preprocessed/Clahe2/Covid"
multiclassifier_healthy_path_train_Clahe2_preprocessed    = "Dataset/Train/Preprocessed/Clahe2/Healthy"
multiclassifier_noncovid_path_train_Clahe2_preprocessed   = "Dataset/Train/Preprocessed/Clahe2/Non-Covid"
###############################################################################################################################
## PRE-PROCESSED-Clahe >> 2.0 << TEST DATASET EXPORT LOCATION ######################################################################################
multiclassifier_covid_path_test_Clahe2_preprocessed      = "Dataset/Test/Preprocessed/Clahe2/Covid"
multiclassifier_healthy_path_test_Clahe2_preprocessed    = "Dataset/Test/Preprocessed/Clahe2/Healthy"
multiclassifier_noncovid_path_test_Clahe2_preprocessed   = "Dataset/Test/Preprocessed/Clahe2/Non-Covid"
###############################################################################################################################
## PRE-PROCESSED-Clahe >> 3.0 << TRAIN ATASET EXPORT LOCATION ######################################################################################
multiclassifier_covid_path_train_Clahe3_preprocessed      = "Dataset/Train/Preprocessed/Clahe3/Covid"
multiclassifier_healthy_path_train_Clahe3_preprocessed    = "Dataset/Train/Preprocessed/Clahe3/Healthy"
multiclassifier_noncovid_path_train_Clahe3_preprocessed   = "Dataset/Train/Preprocessed/Clahe3/Non-Covid"
###############################################################################################################################
## PRE-PROCESSED-Clahe >> 3.0 << TEST DATASET EXPORT LOCATION ######################################################################################
multiclassifier_covid_path_test_Clahe3_preprocessed      = "Dataset/Test/Preprocessed/Clahe3/Covid"
multiclassifier_healthy_path_test_Clahe3_preprocessed    = "Dataset/Test/Preprocessed/Clahe3/Healthy"
multiclassifier_noncovid_path_test_Clahe3_preprocessed   = "Dataset/Test/Preprocessed/Clahe3/Non-Covid"
###############################################################################################################################
original_ds_train_paths=[]
original_ds_train_paths.append(multiclassifier_train_covid_path_original)
original_ds_train_paths.append(multiclassifier_train_healthy_path_original)
original_ds_train_paths.append(multiclassifier_train_noncovid_path_original)
###############################################################################################################################

## Preprocess TRAIN data + export to new directories
for original_ds_path in tqdm(original_ds_train_paths):
    for filename in os.listdir(original_ds_path):

        # read in original data:
        filename_concat = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(original_ds_path,filename),cv2.IMREAD_GRAYSCALE)

        # process image + export to correct directory:
        if img is not None:
            #process image:
            # equ   = cv2.equalizeHist(img)                             #hist equalization
            clahe2imgTrain = cv2.createCLAHE(clipLimit=2.0).apply(img)       #CLAHE
            clahe3imgTrain = cv2.createCLAHE(clipLimit=3.0).apply(img)       #CLAHE

            # export to correct directory:
            # hist_filename = filename_concat + "-HIST-PROCESSED.PNG"
            clahe2_filename = filename_concat + "-CLAHE-2-PROCESSED.PNG"
            clahe3_filename = filename_concat + "-CLAHE-3-PROCESSED.PNG"

            if original_ds_path == multiclassifier_train_covid_path_original:
                cv2.imwrite(os.path.join(multiclassifier_covid_path_train_Clahe2_preprocessed , clahe2_filename), clahe2imgTrain)
                cv2.imwrite(os.path.join(multiclassifier_covid_path_train_Clahe3_preprocessed , clahe3_filename), clahe3imgTrain)

            if original_ds_path == multiclassifier_train_healthy_path_original:
                cv2.imwrite(os.path.join(multiclassifier_healthy_path_train_Clahe2_preprocessed , clahe2_filename), clahe2imgTrain)
                cv2.imwrite(os.path.join(multiclassifier_healthy_path_train_Clahe3_preprocessed , clahe3_filename), clahe3imgTrain)

            if original_ds_path == multiclassifier_train_noncovid_path_original:
                cv2.imwrite(os.path.join(multiclassifier_noncovid_path_train_Clahe2_preprocessed , clahe2_filename), clahe2imgTrain)
                cv2.imwrite(os.path.join(multiclassifier_noncovid_path_train_Clahe3_preprocessed , clahe3_filename), clahe3imgTrain)

        else:
            print("imread failed")

###############################################################################################################################


###############################################################################################################################
original_ds_test_paths=[]
original_ds_test_paths.append(multiclassifier_test_covid_path_original)
original_ds_test_paths.append(multiclassifier_test_healthy_path_original)
original_ds_test_paths.append(multiclassifier_test_noncovid_path_original)
###############################################################################################################################


## Preprocess TEST data + export to new directories
for original_ds_path in tqdm(original_ds_test_paths):
    for filename in os.listdir(original_ds_path):

        # read in original data:
        filename_concat = os.path.splitext(filename)[0]
        imgTest = cv2.imread(os.path.join(original_ds_path,filename),cv2.IMREAD_GRAYSCALE)

        # process image + export to correct directory:
        if img is not None:
            # process image:
            # equ   = cv2.equalizeHist(img)                             #hist equalization
            clahe2imgTest = cv2.createCLAHE(clipLimit=2.0).apply(img)       #CLAHE
            clahe3imgTest = cv2.createCLAHE(clipLimit=3.0).apply(img)       #CLAHE

            # export to correct directory:
            # hist_filename = filename_concat + "-HIST-PROCESSED.PNG"
            clahe2_filename = filename_concat + "-CLAHE-2-PROCESSED.PNG"
            clahe3_filename = filename_concat + "-CLAHE-3-PROCESSED.PNG"

            if original_ds_path == multiclassifier_test_covid_path_original:
                cv2.imwrite(os.path.join(multiclassifier_covid_path_test_Clahe2_preprocessed , clahe2_filename), clahe2imgTest)
                cv2.imwrite(os.path.join(multiclassifier_covid_path_test_Clahe3_preprocessed , clahe3_filename), clahe3imgTest)

            if original_ds_path == multiclassifier_test_healthy_path_original:
                cv2.imwrite(os.path.join(multiclassifier_healthy_path_test_Clahe2_preprocessed , clahe2_filename), clahe2imgTest)
                cv2.imwrite(os.path.join(multiclassifier_healthy_path_test_Clahe3_preprocessed , clahe3_filename), clahe3imgTest)

            if original_ds_path == multiclassifier_test_noncovid_path_original:
                cv2.imwrite(os.path.join(multiclassifier_noncovid_path_test_Clahe2_preprocessed , clahe2_filename), clahe2imgTest)
                cv2.imwrite(os.path.join(multiclassifier_noncovid_path_test_Clahe3_preprocessed , clahe3_filename), clahe3imgTest)

        else:
            print("imread failed")

###############################################################################################################################
