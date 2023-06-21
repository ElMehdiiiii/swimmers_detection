#importing the libraries 
import cv2
import numpy as np

#reading the image
img = cv2.imread("ropes_image.png")

#kernel_size = (3, 3)
#blurred_image = cv2.GaussianBlur(img, kernel_size, 0)

#using canny edge detection, Hough transform will be applied on the output #of cv2.Canny()
cannyedges = cv2.Canny(img, 25, 250)
cv2.imshow("linesDetectedusingHoughTransform", cannyedges)
cv2.waitKey(0)

#applying cv2.HoughlinesP(), the coordinates of the endpoints of the #detected lines are stored
# TODO : parameters to be adjusted also here
detectedlines = cv2.HoughLinesP(cannyedges, 1, np.pi/180, 350,maxLineGap=150)
print(len(detectedlines))
#iterating over the points and drawing lines on the image by using the #coordinates that we got from HoughLinesP()
for line in detectedlines:
  print(line)
  x0, y0, x1, y1 = line[0]
  cv2.line(img, (x0, y0), (x1, y1), (0, 0, 250), 1)

#getting the output
cv2.imshow("linesDetectedusingHoughTransform", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
