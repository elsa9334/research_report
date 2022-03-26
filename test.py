import cv2 as cv
import numpy as np

#read the image
img = cv.imread("test.PNG")

#convert the BGR image to HSV colour space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#set the lower and upper bounds for the green hue
lower_green = np.array([170,50,90])
upper_green = np.array([180, 255, 255])

#create a mask for green colour using inRange function
mask = cv.inRange(hsv, lower_green, upper_green)

#perform bitwise and on the original image arrays using the mask
res = cv.bitwise_and(img, img, mask=mask)

#create resizable windows for displaying the images
cv.namedWindow("res", cv.WINDOW_NORMAL)
cv.namedWindow("hsv", cv.WINDOW_NORMAL)
cv.namedWindow("mask", cv.WINDOW_NORMAL)

#display the images
cv.imshow("mask", mask)
cv.imshow("hsv", hsv)
cv.imshow("res", res)

if cv.waitKey(0):
    cv.destroyAllWindows()