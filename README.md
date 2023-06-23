# Swimmers detection
The Swimmer detection project aims to detect swimmers in competition videos despite the difficulties mainly due to the aquatic environment such as reflections.

## Swimmer detection process
This process consists of three phases using the files below:

1- ***dominated_color_pool.py***: Pre-processing step defining the dominant among the blue color range in the captured video, which will help in the following pool extraction step. 

2- ***poolextract.py***: Pre-processing step: extracts the pool using the previous code. 

3- ***lane_detection.py*** : Pre-processing step: using a dictionary of possible lane colors, this code extracts the lanes. 

4- ***hough_transformation.py*** : Pre-processing stage: once the lanes have been extracted, this code applies a Hough transformation.

5- ***kalman + Hungarian.py***: Post-processing: applies the kalman filter and the Hungarian algorithm. 

6- ***Swimmer detection_dataset_example*** : is a sample of an augmented data set that we used in our training 

7- ***TrainedWeights.pt*** : the trained weights of our Yolo v5 on the basis of the data set of which we have given a sample ( The model being used here is YOLO V5 that you can access via google Colab : https://colab.research.google.com/drive/1gDZ2xcTOgR39tGGs-EZ6i3RTs16wmzZQ ) 
