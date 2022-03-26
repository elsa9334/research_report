from turtle import width
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

filename = '../template_data/mb_side_dec4_rightfinger.txt'


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
xhat = np.array([0.01, 0])
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

x_std = np.std(x_prime)
y_std = np.std(y_prime)
print(x_std*1000)
print(y_std*1000)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Coordinate Variation in the Medium Bumper, Side Pose")

plt.show()
