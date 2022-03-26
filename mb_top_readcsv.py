import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
""" Reading red """
df = pd.read_csv('mb_top_red.csv',sep=',')
df.head()

data = df.to_numpy()
ncol = int((len(data[0])+1)/3 -1)

""" Comment this portion for yellow only"""
for i in range(0,ncol):
    x = data[:,3*i+1]
    y = data[:,3*i+2]
    nan_array = np.isnan(x)
    not_nan_array = ~ nan_array
    x = x[not_nan_array]
    nan_array = np.isnan(y)
    not_nan_array = ~ nan_array
    y = y[not_nan_array]
    p_test = np.polyfit(x,y,5)
    #ynew = np.polyval(p_test,x)
    #ynew = ynew - np.ones_like(ynew)*ynew[0]
    ynew = y - np.ones_like(y)*y[0]

    x = x/3.69
    ynew = ynew/3.28
    #plt.plot(x,ynew,'r',linewidth=0.5)
    plt.scatter(x,ynew,color='red',s=0.5)#,'b',linewidth=0.5)
print(i)

""" Reading yellow """
df = pd.read_csv('mb_top_yell.csv',sep=',')
df.head()

data = df.to_numpy()
ncol = int((len(data[0])+1)/3)
for i in range(0,ncol):
    x = data[:,3*i+1]
    y = data[:,3*i+2]
    nan_array = np.isnan(x)
    not_nan_array = ~ nan_array
    x = x[not_nan_array]
    nan_array = np.isnan(y)
    not_nan_array = ~ nan_array
    y = y[not_nan_array]
    #p_test = np.polyfit(x,y,9)
    #ynew = np.polyval(p_test,x)
    #ynew = ynew - np.ones_like(ynew)*ynew[0]
    ynew = y - np.ones_like(y)*y[0]

    x = (x+24)/3.69
    ynew = (ynew+45)/3.28
    #plt.plot(x,ynew,'y',linewidth=0.5)
    plt.scatter(x,ynew,color='yellow',s=0.5)
print(i)

""" Graph visuals """
plt.gca().invert_yaxis() # flips the y-axis
plt.xlabel('x')
plt.ylabel('y')

#plt.xticks(np.arange(-21, 7, 2))
#plt.yticks(np.arange(-1, 22, 2))
#plt.legend('Red dot trajectory','Yellow dot trajectory')
plt.title('Trajectory variation in toppling medium bumper, top pose')
plt.show()