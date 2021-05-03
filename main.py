import subprocess
import argparse

def execute(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    print("out:\n", result.stdout)
    print("err:\n", result.stderr)
    
parser = argparse.ArgumentParser()
# Face Extraction Module Arguments
parser.add_argument("--predictor", default='./weights/shape_predictor_68_face_landmarks.dat',help="Predictor Path")
parser.add_argument("--detector", default='./weights/mmod_human_face_detector.dat',help="Detector Path")
parser.add_argument("--dataset", default='D:/Academics/Year4/TS/Data/FakeVideos')
                    # 'D:/Academics/Year4/TS/Data/DFDC/train_sample_videos', help="Path to Dataset")
parser.add_argument("--output", default='D:/Academics/Year4/TS/Data/FakeVideos1')
                    # 'D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth', help="Path to Face extracted output")


args = parser.parse_args()

# Face Extraction Module
command1 = "python LipExtraction.py "+ args.predictor + " " + args.detector + " " + args.dataset + " " + args.output
execute(command1)

# Feature Extraction Using Lip Reading Module
execute("conda init bash")
execute("conda activate tf1")
execute("conda list")

command2 = "python ./LipReading/main.py --lip_model_path ./LipReading/models/lrs2_lip_model --data_path "+args.output+" --datalist"+args.output+"/list.txt --graph_type infer"
result2 = subprocess.run(command2, capture_output=True, text=True)

print(result2.stdout)
print(result2.stderr)

command3 = "python ./concatenate_features.py D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/fake D:/Academics/Year4/TS/Data/DFDC/train_videos_mouth/real"
execute(command3)