import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def findLocalMinimum(errs):
	minList = []
	p = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	d = 0.1
	print('Searching for local minimum...')
	for i in p:
		for j in p:
			if np.around(i+j,decimals = 1) > 1:
				continue
			k = np.around(1-(i+j),decimals = 1)
			nb1 = errs.get((np.around(i-d,decimals = 1),np.around(j+d,decimals = 1),k),float('inf'))
			nb2 = errs.get((np.around(i-d,decimals = 1),j,np.around(k+d,decimals = 1)),float('inf'))
			nb3 = errs.get((np.around(i+d,decimals = 1),np.around(j-d,decimals = 1),k),float('inf'))
			nb4 = errs.get((i,np.around(j-d,decimals = 1),np.around(k+d,decimals = 1)),float('inf'))
			nb5 = errs.get((np.around(i+d,decimals = 1),j,np.around(k-d,decimals = 1)),float('inf'))
			nb6 = errs.get((i,np.around(j+d,decimals = 1),np.around(k-d,decimals = 1)),float('inf'))

			if errs[(i,j,k)] <= nb1 and errs[(i,j,k)] <= nb2 and errs[(i,j,k)] <= nb3 and errs[(i,j,k)] <= nb4 and errs[(i,j,k)] <= nb5 and errs[(i,j,k)] <= nb6:
				minList.append((i,j,k))
				
	return minList


def plotErr(pts,errs,ss,sr):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x = pts[:,0]
	y = pts[:,1]
	z = pts[:,2]
	scat = ax.scatter(x,y,z,c=errs,cmap=cm.coolwarm,linewidth=0,antialiased=False)
	ax.set_xlabel('B')
	ax.set_ylabel('G')
	ax.set_zlabel('R')
	fig.colorbar(scat, shrink=0.5, aspect=5)
	ax.view_init(30, 0)
	plt.savefig('sf_' + str(ss) + '_' + str(sr) + '.png')
	plt.close()

