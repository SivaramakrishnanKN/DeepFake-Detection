# DeepFake-Detection
You can either run the main.py root directory of the project with the required arguments, or you can follow these steps to run only the parts required.

1. Run `./LipExtraction.py <path to predictor> <path to detector> <input dataset> <output path>` to extract the face from each frame and split the video into chunks of 1sec and creates the list.txt.
2. Run the following commands from ".Pipeline1/LipReading/models" -

`wget http://www.robots.ox.ac.uk/~vgg/research/deep_lip_reading/models/lrs2_lip_model.zip && \
unzip lrs2_lip_model.zip && \
rm lrs2_lip_model.zip`

`wget http://www.robots.ox.ac.uk/~vgg/research/deep_lip_reading/models/lrs2_language_model.zip && \
unzip lrs2_language_model.zip && \
rm lrs2_language_model.zip`

3. Set the data_path as the folder where the video is saved and data_list as the location where the list of videos along with their caption is stored and then run the following command. Make sure the environment is running on tensorflow 1.x.

`python ./LipReadingmain.py 
--lip_model_path ./LipReading/models/lrs2_lip_model 
--data_path <directory containing videos> 
--data_list <path to list.txt>  
--graph_type infer`

This will extract the feature vector from the video and save it in the same folder as aaa.mp4.npy

4. Run the following command next -

`python concatenate_features.py <fake_rootdir> <real_rootdir>`

This will run through the folder structure and concatenate all the real and fake features and flatten then into a (512+1)x(rows) sized numpy array.
With each row containing the 512 features and a label denoting which video the features belong to.
A dictionary containing the name of the video and aforementioned label as key is also saved for reference.

5. Finally, the classifier -

`python classifier.py <fake features> <real features> <fake video dict> <real video dict>`

6. To run the Frequency analysis -

`python 1D-Powerspectrum.py <fake_rootdir> <real_rootdir>`


I have provided system arguments to run these scripts, but have provided hard coded paths/arguments commented right below the sys.argv initialisation in the code so that you can edit these directly and try it.
