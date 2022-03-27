#!/usr/bin/python
"""Running in terminal python .\ball_tracking.py --video [filename.mp4]"""
# https://github.com/hemkum/ball-tracking-with-opencv
# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file") #type in command
ap.add_argument("-b", "--buffer", type=int, default=600,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries (in HSV)
red_lb =  (170,50,170) #for sb_side (170,120,50) #(170,50,170) others
red_ub = (180, 255, 255)

"""
red_lb =  (170,230,0)
red_ub = (220,255,255)
"""

yell_lb = (20,160,100) #(30,75,100)sbside #(30,75,140)mbtop #(20,130,100)others
yell_ub = (45,255,255)

lower_bound = red_lb
upper_bound = red_ub
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, it exits
if not args.get("video", True):
	print("No video file supplied")
	exit()

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

#Creating a Pandas DataFrame To Store Data Point
Data_Features = ['x', 'y', 'time']
Data_Points = pd.DataFrame(data = None, columns = Data_Features , dtype = float)

#Reading the time in the begining of the video.
start = time.time()

# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()
	
	#Reading The Current Time
	current_time = time.time() - start

	# grabs no frame == end of video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to the HSV color space
	frame = imutils.resize(frame, height=540)
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# construct a mask, then perform a series of dilations and erosions to remove any small blobs left in the mask
	mask = cv2.inRange(hsv, lower_bound, upper_bound)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current (x, y) center of blob
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
	
		# only proceed if the radius meets a minimum size
		if (radius < 300) & (radius > 0.5 ) : 
			# draw the circle and centroid on the frame, then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 3, (0, 0, 255), -1)

			#Save The Data Points
			Data_Points.loc[Data_Points.size/3] = [x , y, current_time]

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and draw the connecting lines
		thickness = 2#int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

#Applying the correction terms to obtain actual experimental data
Data_Points['x'] = Data_Points['x']
Data_Points['y'] = Data_Points['y']
Data_Points['time'] = Data_Points['time']

# Convert to numpy, because I don't want to deal with data in panda
new = Data_Points.to_numpy()
newx = new[:,0]
newy = new[:,1]

# Mask x axis
# threshold = 0 # can change threshold
# diff = np.empty(newx.shape)
# diff[0] = np.inf  # always retain the 1st element
# diff[1:] = np.diff(newx)
# mask = diff > threshold
# newx = newx[mask]
# newy = newy[mask]

""" To find offset distance"""
print(newx[0])
print('yval')
print(newy[0])

# Have all trajectories start at origin
newx = newx - np.ones_like(newx)*newx[0]
newy = newy - np.ones_like(newy)*newy[0]

# Scatter plot
scatter = plt.scatter(newx,newy)
#ysmoothed = gaussian_filter1d(newy, sigma=1)#plt.plot(X_, Y_)
plt.plot(newx, newy)

# Convert filtered to panda
df = pd.DataFrame({'x':newx})
df['y'] = newy
df.to_csv('Filtered.csv', sep=",")


plt.gca().invert_yaxis() # flips the y-axis
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('xy_coords.png')
plt.show()

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
