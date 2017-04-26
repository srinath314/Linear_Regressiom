from numpy import *
import numpy as np
import matplotlib.pyplot as plt

#load some random data. This is the data we use to train the model
x,y1 = loadtxt("book.txt", unpack=True)

#findout length of the vector in data
Nx=x.size

#initialise m, c of the line we want to find using linear regression
m=0
c=1.1

#define learning rate
Lr=0.005

#Here we are using y1 as noise and is added to the random data imported
y=2*x+c+y1




for i in range(20000):

        #find out cost function to be minimised
        J_vect=(1/(2*Nx))*np.square(y-m*x-c)
        J=np.sum(J_vect)

        #find out gradient with respect to m
        DJDm_vect=np.multiply(-x, y-m*x-c)
        DJDm=(1/(Nx))*np.sum(DJDm_vect)

        #find out gradient with respect to c
        DJDc_vect= -(y-m*x-c)
        DJDc=(1/(Nx))*np.sum(DJDc_vect)

        #update m and c
        m=m-Lr*DJDm
        c=c-Lr*DJDc


# print the cost value after minimisation
print(J)

#plot the saple learning data and regression line
plt.plot(x, y,'o',x, m*x+c, 'ro')
plt.show()
