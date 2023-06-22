# Swimmers detection
The Swimmer detection project aims to detect swimmers in competition videos despite the difficulties mainly due to the aquatic environment such as reflections.

## Swimmer detection process
This process consists of three phases using the files below:

1- ***dominated_color_pool.py***: Pre-processing step defining the dominant among the blue color range in the captured video, which will help in the following pool extraction step. 

2- ***poolextract.py***: Pre-processing step: extracts the pool using the previous code. 

3- ***lane_detection.py*** : Pre-processing step: using a dictionary of possible lane colors, this code extracts the lanes. 

4- ***hough_transformation.py*** : Pre-processing stage: once the lanes have been extracted, this code applies a Hough transformation.

5- ***kalman + Hungarian.py***: Post-processing: applies the kalman filter and the Hungarian algorithm. 
