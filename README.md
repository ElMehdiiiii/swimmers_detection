**Swimmers_detection

Project Annexe :

1-dominated_color_pool.py : Pre-Processing step defining the dominant amoung the interval of blue color in the captured video this will help in the next step of pool extraction 

2-poolextract.py : Pre-processing step: that extract the pool thanks to the previous code 

3-lane_detection.py : Pre-processing step : thanks to a dictionary of possible colors of lanes , this code extract the lanes 

4-hough_transformation.py : Pre-processing step : after the extraction of lanes , this code make a hough transformation

5- kalman + Hungarian.py : Post-processing : the kalman filter and the hungarian algorithm 
