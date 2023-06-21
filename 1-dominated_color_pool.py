import cv2
import numpy as np
import matplotlib.pyplot as plt


#def calculate_dominant_color(image):
    # Convert HSV image to RGB
 #   rgb_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    # Threshold to filter out black areas
  #  lower_black = np.array([1, 10, 10], dtype=np.uint8)
   # upper_black = np.array([180, 255, 30], dtype=np.uint8)  # Adjust the threshold if needed
    #black_mask = cv2.inRange(rgb_image, lower_black, upper_black)

    # Apply the mask to the RGB image
   # filtered_image = cv2.bitwise_and(rgb_image, rgb_image, mask=black_mask)

    # Reshape the image to a list of pixels
    #reshaped_image = filtered_image.reshape((-1, 3))

    # Calculate the dominant color
    #dominant_color = np.argmax(np.bincount(reshaped_image[:, 0]))

    #return dominant_color

# Load your HSV image

image = cv2.imread('segmented_image.PNG')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
plt.imshow(image)
plt.show()
np.set_printoptions(threshold=np.inf)

print(image)
plt.imshow(hsv_image)
plt.show()
#print("Image shape:", hsv_image.shape)



# Calculate the dominant color
#dominant_color = calculate_dominant_color(hsv_image)

#print("Dominant color:", dominant_color)

