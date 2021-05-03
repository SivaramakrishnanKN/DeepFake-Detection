import os
import numpy as np
import sys
import pickle
import random
import pandas as pd

#Fake Feature Set
# fake_rootdir = "D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/fake"
fake_rootdir = sys.argv[1] + "/fake"
count_fake=0

fake_videos = {}
for subdir, dirs, files in os.walk(fake_rootdir):
    for file in files: 
        if file.endswith(".npy"):
           count_fake+=1
           if file[:-4] not in fake_videos.keys():
               fake_videos[file[:-4]] = len(fake_videos)+1

fake_dataset = np.zeros((count_fake,30,513))
print("Fake:",count_fake)

count=0
for subdir, dirs, files in os.walk(fake_rootdir):
    for file in files: 
        if file.endswith(".npy"):
            arr = np.full((30,1), fake_videos[file[:-4]])
            temp = np.load(os.path.join(subdir,file))[0]
            fake_dataset[count] = np.concatenate((temp,arr), axis=1)
            count+=1

new_fake = fake_dataset.reshape((fake_dataset.shape[0]*fake_dataset.shape[1]),fake_dataset.shape[2])


#Real Feature Set
# real_rootdir = "D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/real"
real_rootdir = sys.argv[1] + "/real"
count_real=0

real_videos = {}
for subdir, dirs, files in os.walk(real_rootdir):
    for file in files: 
        if file.endswith(".npy"):
           count_real+=1
           if file[:-4] not in real_videos.keys():
               real_videos[file[:-4]] = len(real_videos)+1+len(fake_videos)

real_dataset = np.zeros((count_real,30,513))
print("Real:",count_real)

count=0
for subdir, dirs, files in os.walk(real_rootdir):
    for file in files: 
        if file.endswith(".npy"):
            arr = np.full((30,1), real_videos[file[:-4]])
            temp = np.load(os.path.join(subdir,file))[0]
            real_dataset[count] = np.concatenate((temp,arr), axis=1)
            count+=1

new_real = real_dataset.reshape((real_dataset.shape[0]*real_dataset.shape[1]),real_dataset.shape[2])




np.save(sys.argv[1]+"/fake_features.npy",new_fake)
np.save(sys.argv[1]+"/real_features.npy",new_real)

# np.save("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\fake_features.npy",new_fake)
# np.save("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\real_features.npy",new_real)

try:
    # f = open("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\fake_videos","wb")
    f = open(sys.argv[1]+"fake_videos","wb")
    pickle.dump(fake_videos, f)
    f.close()
    # f = open("D:\Academics\Year4\TS\Data\DFDC\\train_videos_mouth\\real_videos","wb")
    f = open(sys.argv[1]+"fake_videos","wb")
    pickle.dump(real_videos, f)
    f.close()
except:
    print("Smtn went wrong in writing dict")