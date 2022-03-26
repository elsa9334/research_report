import cv2
from numpy.linalg.linalg import matrix_power
from numpy.testing._private.utils import integer_repr
import yaml
import numpy as np
import matplotlib.pyplot as plt
import rospkg
import os
import matplotlib.transforms as transforms
import pyrender
import copy
import statistics
import math
SCALE = 14.78/(660)
filename = '../template_data/sb_side_dec4.txt'

# Helper functions. 
def template_center(template):
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[1]
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    return np.array((x,y))
def load_img(filename, alpha=None):
    img = cv2.imread(filename)
    center = template_center(img)
    b,g,r = cv2.split(img)
    a = (255 - b)
    if alpha is not None:
        a[a==255] = alpha
    else:
        a[a==255] = 127
        
    bgra= cv2.merge((b, g, r, a))
    return bgra, center
def getTransformMatrix(center, theta, scale=1):
    x = center[0]
    y = center[1]
    c, s = np.cos(theta), np.sin(theta)
    M = np.array(((scale*c, -scale*s, x), 
                  (scale*s,  scale*c, y)))
    return M
def compute_transform(pose, scale=1.):
    center = np.array((pose['x'], pose['y']))
    theta = pose['theta']
    M = getTransformMatrix(center, theta, scale)
    return np.vstack((M, (0,0,1)))

folder = "/home/elsa9334/catkin_ws/src/magna_vision"        

# TODO: This should be computed from the filename loaded.
start_template, start_center = load_img(folder + '/images/sift_templates/small_bumper/small_bumper_4.png', alpha=80)
#start_center = template_center(start_template)
end_template, end_center = load_img(folder + '/images/sift_templates/small_bumper/small_bumper_0.png', alpha=10)
print(end_center)
print(start_center)

# start_translation = np.array((500., 500.)) - SCALE*start_center
with open(filename, 'r') as stream:
    data = yaml.safe_load(stream)

ax = plt.gca()
ax.set_aspect('equal')        # Set aspect ratio

# Plot the start frame
extents = 1800*SCALE
t = np.array(start_center)*SCALE
M = transforms.Affine2D().translate(-t[0],-t[1])

M = M + ax.transData
plt.xlim(-50,50)
plt.ylim(-50,50)

# plot axes
# A coordinate frame defined by its origin & unit vectors
origin = np.array([0, 0])
xhat = np.array([5, 0])
yhat = np.array([0, 5])

theta_mat = []
center_mat = np.empty((0,2))

# Plotting 2 unit vectors
ax.arrow(*origin, *xhat, head_width=1, color='r')
ax.arrow(*origin, *yhat, head_width=1, color='g')
for idx, point in enumerate(data['points']):
    try: 
        start_H = compute_transform(point['start pose'])
        end_H = compute_transform(point['end pose'])
    except:
        import ipdb; ipdb.set_trace()
    H = np.linalg.inv(start_H) @ end_H
    R = H[0:2,0:2]
    T = H[0:2,2]
    theta = np.arctan2(R[1,0], R[0,0])
    x_prime = R[:,0]*5
    y_prime = R[:,1]*5
    origin_prime = T*1000
    # plot the end frame
    t2 = np.array(end_center)*SCALE
    M2 = transforms.Affine2D().translate(-t2[0], -t2[1])
    M3 = transforms.Affine2D().rotate(theta).translate(origin_prime[0],origin_prime[1])
    im2 = ax.imshow(end_template, interpolation='none', origin='lower', extent = [0,extents,0,extents])
    M = M2 + M3 + ax.transData
    im2.set_transform(M)
    # plot the arrows
    plt.arrow(*origin_prime, *x_prime, head_width=1, color='r', alpha=0.1)
    plt.arrow(*origin_prime, *y_prime, head_width=1, color='g', alpha=0.1)

    theta_mat = np.append(theta_mat, math.degrees(theta))
    center_mat = np.append(center_mat, np.array([origin_prime/500]), axis=0) # /500 convert to mm

# mean of data

theta_mean = statistics.mean(theta_mat)
x_unit = (center_mat[:,0])
y_unit = (center_mat[:,1])
x_mean = statistics.mean(x_unit)
y_mean = statistics.mean(y_unit)
print(x_mean)
print(x_unit)

x_err = x_unit - x_mean
y_err = y_unit - y_mean
x_sd = math.sqrt(np.var(x_unit))
y_sd = math.sqrt(np.var(y_unit))

plt.savefig('output.png')
plt.show()

plt.hist(x_err, bins=50, edgecolor="black")
plt.xlabel('x_err (mm), sd: '+str(x_sd))
plt.savefig('x_hist.png')
plt.show()

plt.hist(y_err, bins=50, edgecolor="black")
plt.xlabel('y_err (mm), sd: '+str(y_sd))
plt.savefig('y_hist.png')
plt.show()

plt.hist(theta_mat, bins=25, edgecolor="black")
plt.xlabel('theta (deg)')
plt.savefig('theta_hist.png')
plt.show()

