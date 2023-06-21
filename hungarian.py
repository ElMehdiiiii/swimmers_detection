import numpy as np
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

def track_objects(predicted_points, maximum_distance_threshold, maximum_frames_without_detection):
    closed_tracks = []
    open_tracks = []

    for frame, predicted_frame_points in enumerate(predicted_points):
        estimated_points = []  # estimated next points from open tracks obtained by Kalman filter
        for track in open_tracks:
            estimated_point = track.estimate_next_point()  # replace with your Kalman filter estimation
            estimated_points.append(estimated_point)

        associations = associate_points(predicted_frame_points, estimated_points, maximum_distance_threshold)

        for association in associations:
            predicted_index, estimated_index = association
            predicted_point = predicted_frame_points[predicted_index]
            estimated_point = estimated_points[estimated_index]
            open_tracks[estimated_index].add_point(frame, predicted_point, estimated_point)

        for track in open_tracks:
            if track.frames_without_detection > maximum_frames_without_detection:
                closed_tracks.append(track)
            else:
                track.frames_without_detection += 1

        open_tracks = [track for track in open_tracks if track.frames_without_detection <= maximum_frames_without_detection]

        for predicted_index, predicted_point in enumerate(predicted_frame_points):
            is_already_accounted = any(predicted_point in track.predicted_points for track in open_tracks)
            is_already_accounted |= any(predicted_point in track.predicted_points for track in closed_tracks)
            if not is_already_accounted:
                new_track = Track()
                new_track.add_point(frame, predicted_point, None)
                open_tracks.append(new_track)

    closed_tracks += open_tracks

    return closed_tracks
class Track:
    def __init__(self):
        self.predicted_points = []
        self.estimated_points = []
        self.frames_without_detection = 0

    def add_point(self, frame, predicted_point, estimated_point):
        self.predicted_points.append((frame, predicted_point))
        self.estimated_points.append(estimated_point)
        self.frames_without_detection = 0

    def estimate_next_point(self):
        # Implement your Kalman filter estimation here
        # Return the estimated next point
        pass

predicted_points = [
    np.array([[10, 10], [20, 20], [30, 30]]),
    np.array([[15, 15], [25, 25], [35, 35], [45, 45]]),
    np.array([[20, 20], [30, 30], [40, 40]])
]

maximum_distance_threshold = 5
maximum_frames_without_detection = 2

tracks = track_objects(predicted_points, maximum_distance_threshold, maximum_frames_without_detection)
