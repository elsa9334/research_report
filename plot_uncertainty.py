from statistics import mean
from turtle import width
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import math


angles_mat = []
x_mat = []
y_mat = []
plt.rcParams.update({'font.size': 15})

filename = '../template_data/sb_side_dec4.txt'


with open(filename, 'r') as stream:
    data = yaml.safe_load(stream)
    
    
def compute_transform(pose):
    x = pose['x']
    y = pose['y']
    theta = pose['theta']
    
    c, s = np.cos(theta), np.sin(theta)
    H = np.array(((c, -s, x), 
                  (s,  c, y),
                  (0,  0, 1)))
    return H
    

#Boilerplate
plt.gca().set_aspect('equal')        # Set aspect ratio
plt.xlim(-0.01, 0.03)                    # Set x-axis range 
plt.ylim(-0.02, 0.02)                    # Set y-axis range

# A coordinate frame defined by its origin & unit vectors
origin = np.array([0, 0])
xhat = np.array([-0.01, 0])
yhat = np.array([0, -0.01])

# Plotting 2 unit vectors
plt.arrow(*origin, *xhat, width=0.0001, head_width=0.001, color='b')
plt.arrow(*origin, *yhat, width=0.0001,head_width=0.001, color='b')

for idx, point in enumerate(data['points']):
    try: 
        start_H = compute_transform(point['start pose'])
        end_H = compute_transform(point['end pose'])
    except:
        import ipdb; ipdb.set_trace()
    H = np.linalg.inv(start_H) @ end_H
    R = H[0:2,0:2]
    T = H[0:2,2]
    x_prime = R[:,0]*0.01
    y_prime = R[:,1]*0.01
    origin_prime = T
    plt.arrow(*origin_prime, *x_prime, width=0.0001, head_width=0.001, color='r', alpha=0.1)
    plt.arrow(*origin_prime, *y_prime, width=0.0001, head_width=0.001, color='g', alpha=0.1)

    # Standard Dev and other statistics
    angles = np.arctan2(R[1,0], R[0,0])
    #if angles>0:
    #    angles = angles*-1
    angles_mat = np.append(angles_mat, math.degrees(angles))
    x_mat = np.append(x_mat, origin_prime[0])
    y_mat = np.append(y_mat, origin_prime[1])

print('x st dev: (mm)',np.std(x_mat)*1000)
print('x min, max, mean (mm):', min(x_mat),max(x_mat),mean(x_mat))
print('y st dev: (mm)',np.std(y_mat)*1000)
print('y min, max, mean (mm):', min(y_mat),max(y_mat),mean(y_mat))

print('angle st dev: (deg',np.std(angles_mat))
print('angle min, max, mean (deg):', min(angles_mat),max(angles_mat),mean(angles_mat))

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Coordinate Variation in the Medium Bumper, Side Pose")

plt.show()
