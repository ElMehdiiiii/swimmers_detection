import cv2
import numpy as np

color_dict = {
    'blue': ([101, 100, 70], [134, 255, 255]),
    'yellow': ([10, 0, 0], [80, 255, 255]),
    'red1': ([0, 50, 50], [10, 255, 255]),
    'red2': ([130, 0, 0], [180, 255, 255])
}

image = cv2.imread('left.PNG')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

ropes_mask = np.zeros(image.shape[:2], dtype=np.uint8)

for color, (lower_range, upper_range) in color_dict.items():
    mask = cv2.inRange(hsv_image, np.array(lower_range), np.array(upper_range))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    ropes_mask = cv2.bitwise_or(ropes_mask, mask)

ropes = cv2.bitwise_and(image, image, mask=ropes_mask)
cv2.imshow('ropes', ropes)

cv2.imwrite('ropes_image.png', ropes)

cv2.waitKey(0)
cv2.destroyAllWindows()

# TODO :  deal with discontinuities can solve the problem in the hough_transformation step
