import numpy as np
import numpy.linalg as LA
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    A = np.zeros((2*N, 8))
	# if you take solution 2:
	# A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    # TODO: compute H from A and b
    for i in range(N):
        ux = u[i,0]
        uy = u[i,1]
        vx = v[i,0]
        vy = v[i,1]
        A[2*i,:] = np.array([ux,uy,1,0,0,0,-ux*vx,-uy*vx])
        A[2*i+1,:] = np.array([0,0,0,ux,uy,1,-ux*vy,-uy*vy])
    b = v.reshape(-1,1)
    H = LA.solve(A,b)
    H = np.concatenate((H,np.array([[1]])))
    H = H.reshape(3,3)
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic
    p = np.array([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
    H = solve_homography(p,corners)
    for y in range(h):
        for x in range(w):
            newPoint = H @ (np.array([[x,y,1]])).T
            newPoint = np.around(newPoint / newPoint[2])
            canvas[int(newPoint[1]),int(newPoint[0]),:] = img[y,x,:]
    return canvas


def backWarp(img, canvas, corners):
    h, w, ch = img.shape
    p = np.array([[0,0], [w-1,0], [0,h-1], [w-1,h-1]])
    H = solve_homography(p,corners)
    for y in range(h):
        for x in range(w):
            newPoint = H @ (np.array([[x,y,1]])).T
            newPoint = np.around(newPoint / newPoint[2])
            img[y,x,:] = canvas[int(newPoint[1]),int(newPoint[0]),:]
    return img


def main():
    # Part 1
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/wu.jpg')
    img2 = cv2.imread('./input/ding.jpg')
    img3 = cv2.imread('./input/yao.jpg')
    img4 = cv2.imread('./input/kp.jpg')
    img5 = cv2.imread('./input/lee.jpg')

    corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]])
    corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
    corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
    corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
    corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])

    # TODO: some magic
    canvas = transform(img1,canvas,corners1)
    canvas = transform(img2,canvas,corners2)
    canvas = transform(img3,canvas,corners3)
    canvas = transform(img4,canvas,corners4)
    canvas = transform(img5,canvas,corners5)
    cv2.imwrite('part1.png', canvas)

    # Part 2
    img = cv2.imread('./input/screen.jpg')
    # TODO: some magic
    corners = np.array([[1041,370], [1100,396], [984,552], [1036,599]])
    size = 150
    output2 = np.zeros((size,size,3))
    output2 = backWarp(output2,img,corners)
    cv2.imwrite('part2.png', output2)

    # Part 3
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    # TODO: some magic
    corners = np.array([[160,129], [563,129], [0,286], [723,286]])
    size = 400
    output3 = np.zeros((size,size,3))
    output3 = backWarp(output3,img_front,corners)
    cv2.imwrite('part3.png', output3)


if __name__ == '__main__':
    main()
