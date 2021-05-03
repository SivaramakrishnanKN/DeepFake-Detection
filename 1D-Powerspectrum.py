import cv2
import numpy as np
import os
import radialProfile
import glob
from matplotlib import pyplot as plt
import pickle
import time
from scipy.interpolate import griddata
import sys

count_real = 0
count_fake = 0

# rootdir =  'D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/fake'
rootdir = sys.argv[1]

for subdir, dirs, files in os.walk(rootdir):
    for file in files:        
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            count_fake+=1

# rootdir2 = 'D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/real'
rootdir2 = sys.argv[2]

for subdir, dirs, files in os.walk(rootdir2):
    for file in files:        
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            count_real+=1
            
number_iter = min(count_fake, count_real)
print(number_iter)

data= {}
epsilon = 1e-8
N = 300
y = []
error = []

# number_iter = min(count_fake, count_real)
# print(number_iter)
# number_iter = 85382

psd1D_total = np.zeros([number_iter, N])
label_total = np.zeros([number_iter])
psd1D_org_mean = np.zeros(N)
psd1D_org_std = np.zeros(N)


cont = 0
count_real = 0
count_fake = 0

start = time.time()
#fake data
# rootdir =  'D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/fake'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:        
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            count_fake+=1
            filename = os.path.join(subdir, file)
            
            img = cv2.imread(filename,0)

            # we crop the center
            h = int(img.shape[0]/3)
            w = int(img.shape[1]/3)
            img = img[h:-h,w:-w]

            # Discrete Fourier Transform
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += epsilon

            magnitude_spectrum = 20*np.log(np.abs(fshift))
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            # Calculate the azimuthally averaged 1D power spectrum
            points = np.linspace(0,N,num=psd1D.size) # coordinates of a
            xi = np.linspace(0,N,num=N) # coordinates for interpolation

            interpolated = griddata(points,psd1D,xi,method='cubic')
            interpolated /= interpolated[0]

            psd1D_total[cont,:] = interpolated             
            label_total[cont] = 0
            cont+=1
            print(filename, h, w)

        if cont == number_iter:
            break
    if cont == number_iter:
        break
            
for x in range(N):
    psd1D_org_mean[x] = np.mean(psd1D_total[:,x])
    psd1D_org_std[x]= np.std(psd1D_total[:,x])


## real data
psd1D_total2 = np.zeros([number_iter, N])
label_total2 = np.zeros([number_iter])
psd1D_org_mean2 = np.zeros(N)
psd1D_org_std2 = np.zeros(N)


cont = 0
# rootdir2 = 'D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/real'

for subdir, dirs, files in os.walk(rootdir2):
    for file in files:        
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            count_real+=1
            filename = os.path.join(subdir, file)
            parts = filename.split("/")

            img = cv2.imread(filename,0)

            # we crop the center
            h = int(img.shape[0]/3)
            w = int(img.shape[1]/3)
            img = img[h:-h,w:-w]

            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            fshift += epsilon


            magnitude_spectrum = 20*np.log(np.abs(fshift))

            # Calculate the azimuthally averaged 1D power spectrum
            psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)

            points = np.linspace(0,N,num=psd1D.size) # coordinates of a
            xi = np.linspace(0,N,num=N) # coordinates for interpolation

            interpolated = griddata(points,psd1D,xi,method='cubic')
            interpolated /= interpolated[0]

            psd1D_total2[cont,:] = interpolated             
            label_total2[cont] = 1
            cont+=1
            print(filename, h, w)


            
        if cont == number_iter:
            break
    if cont == number_iter:
        break

print("Fake:", count_fake, "Real:", count_real)
        
for x in range(N):
    psd1D_org_mean2[x] = np.mean(psd1D_total2[:,x])
    psd1D_org_std2[x]= np.std(psd1D_total2[:,x])
    
    
y.append(psd1D_org_mean)
y.append(psd1D_org_mean2)

error.append(psd1D_org_std)
error.append(psd1D_org_std2)

psd1D_total_final = np.concatenate((psd1D_total,psd1D_total2), axis=0)
label_total_final = np.concatenate((label_total,label_total2), axis=0)

data["data"] = psd1D_total_final
data["label"] = label_total_final

output = open('data.pkl', 'wb')
pickle.dump(data, output)
output.close()
print("Execution time:", time.time()-start)
print("DATA Saved") 

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, auc, roc_auc_score
from tensorflow.keras.models import model_from_json
from tensorflow.keras.metrics import AUC

#train
pkl_file = open('data.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

X = data["data"]
y = data["label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

logreg = LogisticRegression(solver='liblinear', max_iter=1000)
logreg.fit(X_train, y_train)

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

# Train the models
logreg_pred = logreg.predict(X_test)
print("Accuracy:",accuracy_score(y_test, logreg_pred))
print("F1 score macro:",f1_score(y_test, logreg_pred, average='binary'))
print("Confusion Matrix", confusion_matrix(y_test, logreg_pred))
print("AUC", roc_auc_score(y_test, logreg_pred))


# save the model to disk
filename = 'logreg_model.sav'
pickle.dump(logreg, open(filename, 'wb'))
print("Saved model to disk")  


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

svclassifier_pred = svclassifier.predict(X_test)
print("Accuracy:",accuracy_score(y_test, svclassifier_pred))
print("F1 score macro:",f1_score(y_test, svclassifier_pred, average='binary'))
print("Confusion Matrix", confusion_matrix(y_test, svclassifier_pred))
print("AUC", roc_auc_score(y_test, svclassifier_pred))

# save the model to disk
filename = 'svclassifier_model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))
print("Saved model to disk")  


svclassifier_r = SVC(C=6.37, kernel='rbf', gamma=0.86)
svclassifier_r.fit(X_train, y_train)


svclassifier_r_pred = svclassifier_r.predict(X_test)
print("Accuracy:",accuracy_score(y_test, svclassifier_r_pred))
print("F1 score macro:",f1_score(y_test, svclassifier_r_pred, average='binary'))
print("Confusion Matrix", confusion_matrix(y_test, svclassifier_r_pred))
print("AUC", roc_auc_score(y_test, svclassifier_r_pred))

# save the model to disk
filename = 'svclassifier_r_model.sav'
pickle.dump(svclassifier_r, open(filename, 'wb'))
print("Saved model to disk")

svclassifier_p = SVC(C=6.37, kernel='rbf', gamma=0.86)
svclassifier_p.fit(X_train, y_train)

svclassifier_p_pred = svclassifier_p.predict(X_test)
print("Accuracy:",accuracy_score(y_test, svclassifier_p_pred))
print("F1 score macro:",f1_score(y_test, svclassifier_p_pred, average='binary'))
print("Confusion Matrix", confusion_matrix(y_test, svclassifier_p_pred))
print("AUC", roc_auc_score(y_test, svclassifier_p_pred))

# save the model to disk
filename = 'svclassifier_p_model.sav'
pickle.dump(svclassifier_p, open(filename, 'wb'))
print("Saved model to disk")