import numpy as np
import time
from tensorflow.keras.models import Sequential
# from tensorflow.keras import metrics
from tensorflow.keras.layers import Dense
import pandas as pd
import pickle 
import random
import os
import sys

'''
Initialize variables

fake - all concatenated fake feature vectors
real - all concatenated real feature vetors

fake_videos - dictionary containing the mapping of each feature vector to the corresponding fake video
real_videos - dictionary containing the mapping of each feature vector to the corresponding real video

X - set of concatenated fake and real features
y - set of concatenated labels containing 1 for fake and 0 for real
labels - label containing the key of which video the frame belongs to

'''

# fake = np.load("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\fake_features.npy")
fake = np.load(sys.argv[1])
# real = np.load("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\real_features.npy")
real = np.load(sys.argv[2])

# fake_videos = pickle.load(open("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\fake_videos","rb"))
fake_videos = pickle.load(sys.argv[3])
# real_videos = pickle.load(open("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\real_videos","rb"))
real_videos = pickle.load(sys.argv[4])
total_videos = len(fake_videos)+len(real_videos)
classes = np.array(['Real','Fake'])

X = pd.DataFrame(fake[:,:-1]).append(pd.DataFrame(real[:,:-1]), ignore_index=True)
y = pd.DataFrame(np.ones(fake.shape[0])).append(pd.DataFrame(np.zeros(real.shape[0])), ignore_index =True)
labels = pd.DataFrame(fake[:,-1:]).append(pd.DataFrame(real[:,-1:]), ignore_index=True)

# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_auc_score
from tensorflow.keras.models import model_from_json
from tensorflow.keras.metrics import AUC, f1_score

'''
Split the dataset into the train and test set

You can comment out this part if the data is already split
'''
trainsample = random.sample(range(1,total_videos+1), int(total_videos*0.8))
trainlist = {i:(1 if i in trainsample else 0) for i in range(1,total_videos+1)}
# testlist = [i for i in range(1,total_videos) if i not in trainlist]

X_train = pd.DataFrame(columns=range(X.shape[1]))
y_train = pd.DataFrame(columns=[0])
labels_train = pd.DataFrame(columns=[0])
X_test = pd.DataFrame(columns=range(X.shape[1]))
y_test = pd.DataFrame(columns=[0])
labels_test = pd.DataFrame(columns=[0])


t = time.time()
for i in range(len(X)):
    if trainlist[labels.loc[i][0]]==1:
        X_train = X_train.append(X.loc[i], ignore_index=True)
        y_train = y_train.append(y.loc[i], ignore_index=True)
        labels_train = labels_train.append(labels.loc[i], ignore_index=True)
    else:
        X_test = X_test.append(X.loc[i], ignore_index=True)
        y_test = y_test.append(y.loc[i], ignore_index=True)
        labels_test = labels_test.append(labels.loc[i], ignore_index=True)
print("time: ",time.time()-t)
y_train = y_train.drop(columns=[1])
y_test = y_test.drop(columns=[1])
labels_test = labels_test.drop(columns=[1])
labels_train = labels_train.drop(columns=[1])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

'''
Construct the fully connected layer and fit on the data
'''

classifier = Sequential()
classifier.add(Dense(units=512, input_dim=512, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=50, validation_split=0.1)
classifier.evaluate(X_test, y_test)

# Load saved model
# json_file = open('./results/classifier.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(loaded_model_json)
# # load weights into new model
# classifier.load_weights("./resutls/classifier.h5")
# print("Loaded model from disk")

# Save Model
model_json = classifier.to_json()
with open("./results/classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("./results/classifier.h5")
print("Saved model to disk")

'''
Evaluate Frame Level Accuracy
'''
y_pred = classifier.predict(X_test)
y_pred = [0 if y<0.5 else 1 for y in y_pred]


# print(auc(fpr, tpr))
print("Frame Level Results:")
print("Accuracy:",accuracy_score(y_test, y_pred))
print("F1 score macro:",f1_score(y_test, y_pred, average='binary'))
print("Confusion Matrix", confusion_matrix(y_test, y_pred))
print("AUC", roc_auc_score(y_test, y_pred))

'''
Evaluate Video Level Accuracy

Majority vote classifies a given video as fake if more frames from the video have been classified as fake and vice versa
'''
tn = 0 #REAL,REAL
fp = 0 #REAL,FAKE
fn = 0 #FAKE,REAL
tp = 0 #FAKE,FAKE

actual_val = {}
for i in range(len(labels_test)):
    actual_val[labels_test.loc[i][0]] = y_test.loc[i]
fake_preds = dict.fromkeys(actual_val.keys(),0)
real_preds = dict.fromkeys(actual_val.keys(),0)
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        fake_preds[labels_test.loc[i][0]]+=1

    else:
        real_preds[labels_test.loc[i][0]]+=1

y_video_pred = []
y_video_test = []

for i in actual_val.keys():
    if fake_preds[i]>=real_preds[i]:
        y_video_pred.append(1) 
        
        if actual_val[i][0] == 1:
            tp+=1
            y_video_test.append(1) 
        else:
            fp+=1
            y_video_test.append(0) 
    else:
        y_video_pred.append(0) 
        if actual_val[i][0] == 1:
            fn+=1
            y_video_test.append(1) 
        else:
            tn+=1
            y_video_test.append(0) 

print("Video Level Results:")
print("Accuracy:",accuracy_score(y_video_test, y_video_pred))
print("F1 score macro:",f1_score(y_video_test, y_video_pred, average='binary'))
print("Confusion Matrix", confusion_matrix(y_video_test, y_video_pred))
print("AUC", roc_auc_score(y_video_test, y_video_pred))

print(tn, fp)
print(fn, tp)

'''
Single Video Test
'''

# test_vid = np.load("D:/Academics/Year4/TS/Code/deep_lip_reading-dependabot-pip-tensorflow-gpu-2.3.1/media/example/aaaaaa.mp4.npy")[0]
# # test_vid = test_vid.reshape(test_vid.shape[0]*test_vid.shape[1])

# pred = classifier.predict(test_vid)
# pred = [0 if y<0.5 else 1 for y in pred]
# one = 0
# zero = 0
# for i  in pred:
#     if i==0:
#         zero+=1
#     if i==1:
#         one+=1
# print(zero, one)
