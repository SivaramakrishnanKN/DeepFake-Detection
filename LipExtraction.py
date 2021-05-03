import numpy as np
import cv2
import dlib
import math
import sys
import pickle
import argparse
import os
import pandas as pd
import json
import skvideo.io
import time
import shutil
from IPython.display import Video
import sys

# Dlib requirements.
predictor_path = sys.argv[1]
detector_path = sys.argv[2]
detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1(detector_path)
predictor = dlib.shape_predictor(predictor_path)

#Dataset
dataset = sys.argv[3]
output = sys.argv[4]

fake_mouth_frames = output+'/fake'
real_mouth_frames = output+'/real'

if not os.path.exists(real_mouth_frames):
    os.makedirs(real_mouth_frames)
if not os.path.exists(fake_mouth_frames):
    os.makedirs(fake_mouth_frames)

    
# Create a list of all videos part of the training set
train_list = list(os.listdir(dataset))
train_list.remove("metadata.json")

metadata = pd.read_json(dataset+'/metadata.json').T
print(metadata)

t = time.time()

max_frames = 30
min_frames = 30

for video in train_list:
        
    inputparameters = {}
    outputparameters = {}
    reader = skvideo.io.FFmpegReader(os.path.join(dataset,video),
                    inputdict=inputparameters,
                    outputdict=outputparameters)
    
    
    # Create folder to save fake mouth frames and fake full frames
    if metadata.loc[video]["label"] == 'FAKE':
        if not os.path.exists(os.path.join(fake_mouth_frames,video)):
            os.makedirs(os.path.join(fake_mouth_frames,video))

        mouth_path = fake_mouth_frames+'/'+video


    # Create folder to save real mouth frames and real full frames
    if metadata.loc[video]["label"] == "REAL":
        if not os.path.exists(os.path.join(real_mouth_frames,video)):
            os.makedirs(os.path.join(real_mouth_frames,video))
        mouth_path = real_mouth_frames+'/'+video


    video_shape = reader.getShape()
    (num_frames, h, w, c) = video_shape
    print(video,":",num_frames, h, w, c)
    
    # Required parameters for mouth extraction.
    width_crop_max = 0
    height_crop_max = 0
    
    # Initialize Counters
    frame_count=0
    folder_count=0
    count=0
    count_detected=0
    batch_count = 0
    total_count_detected = 0
    t_v = time.time()
        
    # Initialize the writer
    if not os.path.exists(mouth_path+"/"+str(folder_count)):
        os.makedirs(mouth_path+"/"+str(folder_count))
    writer = skvideo.io.FFmpegWriter(mouth_path+"/"+str(folder_count)+"/"+video)
    log = open(mouth_path+"/"+str(folder_count)+"/log.txt", "w")
    print("writing at",mouth_path+"/"+str(folder_count)+"/"+video)
    
    # Iterate through all the frames in the video
    for frame in reader.nextFrame():
        t_f = time.time()
        
        frame_count+=1
        count+=1
        batch_count+=1
        
        # Detection of the frame
        frame.setflags(write=True)
        detections = detector(frame, 1)


        # If the face is not detected log the frame number
        if len(detections) == 0:
            # Set detected flag to False if prev detection was True
            print("No face detected.")
            log.write(str(batch_count)+"\n")

        
        else:
            for k, d in enumerate(detections):
                count_detected+=1
                total_count_detected+=1
                l = d.left()
                t = d.top()
                r = d.right()
                b = d.bottom()
#                 l = max(w,h)
                w = r-l
                h = b-t
                s = min(w,h)
                print(l,r,"/",t,b,s)
#                 face = frame[t:b, l:r, :]
                face = frame[int(t):int(t+s), int(l):int(l+s)]
                face_resized = cv2.resize(face,(160,160),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                print(face.shape, face_resized.shape)
                # Shape of the face.
#                 shape = predictor(frame, d)

                # Save the face as retaining the same resolution
                if not os.path.exists(mouth_path+'/'+str(folder_count)):
                    os.makedirs(mouth_path+'/'+str(folder_count))
                cv2.imwrite(mouth_path+'/'+str(folder_count)+'/'+str(batch_count)+".png", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                
                # Write the resized frame to the video snippet
                writer.writeFrame(face_resized)
                break
        # Split the video into smaller chunks
        print("\nFrame: ",count,"/",num_frames, "Batch Detected: ", count_detected, "/", batch_count)
        if batch_count == max_frames:
            print(count_detected, "/", batch_count,"detected")
            batch_count = 0
            if count_detected==0:
                log.close()
                shutil.rmtree(mouth_path+"/"+str(folder_count))
                if not os.path.exists(mouth_path+"/"+str(folder_count)):
                    os.makedirs(mouth_path+"/"+str(folder_count))
                writer = skvideo.io.FFmpegWriter(mouth_path+"/"+str(folder_count)+"/"+video)
                log = open(mouth_path+"/"+str(folder_count)+"/log.txt", "w")
                print("writing at",mouth_path+"/"+str(folder_count)+"/"+video)
            else:
                count_detected=0
                folder_count += 1
                log.close()
                writer.close()
                if not os.path.exists(mouth_path+"/"+str(folder_count)):
                    os.makedirs(mouth_path+"/"+str(folder_count))
                writer = skvideo.io.FFmpegWriter(mouth_path+"/"+str(folder_count)+"/"+video)
                log = open(mouth_path+"/"+str(folder_count)+"/log.txt", "w")
                print("writing at",mouth_path+"/"+str(folder_count)+"/"+video)
                
        # Save the video if its the last frame
        if frame_count==num_frames and count_detected >= min_frames:
            writer.close()
            log.close()

    print(video,": ",time.time()-t_v,"s")
    print("%d of %d frames detected" % (total_count_detected, count))
    
print("Overall Execution Time %d", time.time() - t)

video_list = open(dataset+"/list.txt", "w")
rootdir = dataset
for subdir, dirs, files in os.walk(rootdir):
    for file in files: 
        if file.endswith(".mp4"):
            print(subdir,file)
            string = os.path.join(subdir,file) + ", AND\n"
            video_list.write(string)
video_list.close()