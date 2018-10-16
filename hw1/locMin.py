import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def findLocalMin(errs):
	minList = []
	p = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	d = 0.1
	
	for i in p:
		for j in p:
			k = np.around(1-(i+j),decimals = 1)
			if k < 0:
				continue
			nb1 = errs.get((np.around(i-d,decimals = 1),np.around(j+d,decimals = 1),k),float('inf'))
			nb2 = errs.get((np.around(i-d,decimals = 1),j,np.around(k+d,decimals = 1)),float('inf'))
			nb3 = errs.get((np.around(i+d,decimals = 1),np.around(j-d,decimals = 1),k),float('inf'))
			nb4 = errs.get((i,np.around(j-d,decimals = 1),np.around(k+d,decimals = 1)),float('inf'))
			nb5 = errs.get((np.around(i+d,decimals = 1),j,np.around(k-d,decimals = 1)),float('inf'))
			nb6 = errs.get((i,np.around(j+d,decimals = 1),np.around(k-d,decimals = 1)),float('inf'))

			if errs[(i,j,k)] < nb1 and errs[(i,j,k)] < nb2 and errs[(i,j,k)] < nb3 and errs[(i,j,k)] < nb4 and errs[(i,j,k)] < nb5 and errs[(i,j,k)] < nb6:
				minList.append((i,j,k))
				
	return minList


