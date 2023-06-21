import cv2 as cv
import numpy as np
import os
from numpy import genfromtxt
from scipy.optimize import linear_sum_assignment

def associate_points(predicted_points, estimated_points, maximum_distance):
    num_predicted = len(predicted_points)
    num_estimated = len(estimated_points)
    distance_matrix = np.zeros((num_predicted, num_estimated))

    for i in range(num_predicted):
        for j in range(num_estimated):
            distance_matrix[i, j] = np.linalg.norm(predicted_points[i] - estimated_points[j])

    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    associations = []

    for row, col in zip(row_indices, col_indices):
        distance = distance_matrix[row, col]
        if distance <= maximum_distance:
            associations.append((row, col))

    return associations

class KalmanFilter:
    def __init__(self, initial_x, initial_y, initial_dx, initial_dy, dt):
        self.kf = cv.KalmanFilter(4, 2)

        initial_state = np.array([[initial_x], [initial_y], [initial_dx], [initial_dy]], dtype=np.float32)  # Initial state estimate
        self.kf.statePost = initial_state

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

        process_noise = 1e-5  # Process noise covariance
        measurement_noise = 1e-1  # Measurement noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]], dtype=np.float32)
        predicted = self.kf.predict()
        if coordX is not None and coordY is not None:
            self.kf.correct(measured)
            estimated_state = self.kf.statePost
        else:
            estimated_state = predicted
        return estimated_state

# Example usage
initial_x1 = 0
initial_y1 = 0
initial_dx1 = 0
initial_dy1 = 0

initial_x2 = 0
initial_y2 = 0
initial_dx2 = 0
initial_dy2 = 0

dt = 0.1

video_file = "video2.mp4"  # Path to your video file
fichier_label = "yolo_results.csv"

# liste des points prise par yolo
predicted_points_all_frames = genfromtxt(fichier_label, delimiter=',')
objets_points_top_left = [[0] for i in range(len(predicted_points_all_frames))] # dictionary of {frame : top left}
objets_points_bottom_right = [[0] for i in range(len(predicted_points_all_frames))] # dictionary of {frame : bottom right}
estimated_points_top = [[0,0]]
estimated_points_bottom = [[0,0]]
kalman_filter1 = KalmanFilter(initial_x1, initial_y1, initial_dx1, initial_dy1, dt)  # Kalman filter for top-left points
kalman_filter2 = KalmanFilter(initial_x2, initial_y2, initial_dx2, initial_dy2, dt)  # Kalman filter for bottom-right points

# Example measurement coordinates
coordX1 = 10
coordY1 = 5
coordX2 = 20
coordY2 = 15

if not os.path.exists(fichier_label):
    print("Le fichier de label n'existe pas ...")
    quit()

width = 1200
height = 800

cap = cv.VideoCapture(video_file)  # Open the video file
id_frame = 0
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break
    frame = cv.resize(frame, (width, height))  # Resize the frame
    mask = predicted_points_all_frames[:, 0] == id_frame

    points = []
    if np.any(mask):
        for d in predicted_points_all_frames[mask, :]:
            xm = int(d[1] + d[3] / 2)
            ym = int(d[2] + d[4] / 2)
            cv.circle(frame, (xm, ym), 2, (0, 255, 0), 2)
            points.append([xm, ym, int(d[3]), int(d[4])])
            coordX1, coordY1 = points[-1][0] - points[-1][2]/2, points[-1][1] - points[-1][3]/2
            objets_points_top_left[id_frame].append(coordX1 , coordY1)
            coordX2, coordY2 = points[-1][0] + points[-1][2]/2, points[-1][1] + points[-1][3]/2
            objets_points_bottom_right.append(coordX2 , coordY2)
    else:
        coordX1, coordY1, coordX2, coordY2 = None, None, None, None

    estimated_state1 = kalman_filter1.Estimate(coordX1, coordY1)
    estimated_state2 = kalman_filter2.Estimate(coordX2, coordY2)

    estimated_state1 = [estimated_state1[i][0] for i in range(len(estimated_state1))]
    estimated_state2 = [estimated_state2[i][0] for i in range(len(estimated_state2))]

    
    estimated_points_top.append([estimated_state1[0], estimated_state1[1]])
    estimated_points_bottom.append([ estimated_state2[0], estimated_state2[1]])
    id_frame += 1

print(estimated_points_top)
print(estimated_points_bottom)
cap.release()  # Release the video capture
cv.destroyAllWindows()
