import cv2
import numpy as np


image = cv2.imread('Capture.PNG')
desired_width = 800
desired_height = 600

# Resize the image
resized_image = cv2.resize(image, (desired_width, desired_height))
# Define the water color range (in HSV)
lower_range = np.array([90, 70, 70], dtype=np.uint8) # h_min , s_min , v_min
upper_range = np.array([200, 255, 255], dtype=np.uint8) # h_max , s_max , v_max0
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_range = np.tile(lower_range, (hsv_image.shape[0], hsv_image.shape[1], 1))
upper_range = np.tile(upper_range, (hsv_image.shape[0], hsv_image.shape[1], 1))
# Create a mask based on the color range
mask = cv2.inRange(hsv_image, lower_range, upper_range)
# Apply morphological operations to remove noise and refine the mask
kernel = np.ones((4,4), np.uint8)
# !!!! commentaire : 4 et le nombre d'iterations can be optimized
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=2)

# Apply the mask to the original image
segmented_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite('segmented_image.PNG', segmented_image)

# Display the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

