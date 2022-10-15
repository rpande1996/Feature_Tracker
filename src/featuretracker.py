import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg.linalg import LinAlgError


def readImages(folder, num_images):
    arr_images = []
    for i in range(num_images):
        arr_images.append(cv2.imread(f'{folder}hotel.seq{i}.png'))
    return arr_images


def gray(image):
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gr


def generateGaussian(size, scaleX, scaleY):
    lower_limit = int(-((size - 1) / 2))
    upper_limit = abs(lower_limit) + 1
    ind = np.arange(lower_limit, upper_limit)
    row = np.reshape(ind, (ind.shape[0], 1)) + np.zeros((1, ind.shape[0]))
    col = np.reshape(ind, (1, ind.shape[0])) + np.zeros((ind.shape[0], 1))
    G = (1 / (2 * np.pi * (scaleX * scaleY))) * np.exp(
        -((col ** 2 / (2 * (scaleX ** 2))) + (row ** 2 / (2 * (scaleY ** 2)))))
    return G


def Gblur(size, sig, image):
    kern = generateGaussian(size, sig, sig)
    blur = cv2.filter2D(image, -1, kern)
    return blur


def cornerDisconnect(win):
    new_win = np.zeros(win.shape).astype(np.uint8)
    available_points = []
    new_available = []
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            if win[i, j] > 0:
                available_points.append((i, j))
    distances = []

    if len(available_points) > 0:
        for i in range(len(available_points) - 1):
            distances.append((((available_points[i][0] - available_points[i + 1][0]) ** 2) + (
                    (available_points[i][1] - available_points[i + 1][1]) ** 2)) ** 0.5)
        if np.all(distances) <= np.sqrt(2):
            new_available.append(available_points[0])
        else:
            new_available = available_points

        for i in range(len(new_available)):
            new_win[new_available[i][0], new_available[i][1]] = 1
        return new_win
    else:
        return win


def cornerDetection(image, size, sig, thresh, k, win_size):
    gray_blur = Gblur(size, sig, gray(image))
    Ix, Iy = np.gradient(gray_blur)
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    off = int(win_size / 2)
    x_range = image.shape[0] - off
    y_range = image.shape[1] - off
    img = np.zeros((image.shape[0], image.shape[1]))
    Img = image.copy()
    for x in range(off, x_range):
        for y in range(off, y_range):
            start_x = x - off
            end_x = x + off + 1
            start_y = y - off
            end_y = y + off + 1
            windowIxx = Ixx[start_x: end_x, start_y: end_y]
            windowIxy = Ixy[start_x: end_x, start_y: end_y]
            windowIyy = Iyy[start_x: end_x, start_y: end_y]

            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            r = det - k * (trace ** 2)
            if r > thresh:
                img[x, y] = 255
    non = disconnectNonMax(img, 5, 1)
    keypoint_xcoor = []
    keypoint_ycoor = []
    for i in range(Img.shape[0]):
        for j in range(Img.shape[1]):
            if non[i, j] == 255:
                Img[i, j, 1] = 255
                keypoint_xcoor.append(i)
                keypoint_ycoor.append(j)

    corners_coor = createKeypointList(keypoint_xcoor, keypoint_ycoor)
    return corners_coor


def disconnectNonMax(array1, size, step):
    array = array1.copy()
    for i in range(0, array.shape[0] - size, step):
        for j in range(0, array.shape[1] - size, step):
            win = array[i:i + size, j:j + size]
            new_win = cornerDisconnect(win)
            array[i:i + size, j:j + size] = new_win

    return (array * 255).astype(np.uint8)


def createKeypointList(x_list, y_list):
    list_of_Keypoints = []
    for i in range(len(x_list)):
        list_of_Keypoints.append((x_list[i], y_list[i]))

    return list_of_Keypoints


def trackPoints(x, y, imag1, imag2, win_size):
    imagE1 = gray(imag1)
    imagE2 = gray(imag2)
    im1 = imagE1.astype(float)
    im2 = imagE2.astype(float)
    off = int(win_size / 2)
    u = 0
    v = 0
    i = 1
    while True:
        U = u
        V = v
        image1 = im1[(x + u) - off:(x + u) + off + 1, (y + v) - off:(y + v) + off + 1]
        image2 = im2[(x + u) - off:(x + u) + off + 1, (y + v) - off:(y + v) + off + 1]
        Ix, Iy = np.gradient(im1)
        Ix1 = Ix[x - off:x + off + 1, y - off:y + off + 1]
        Iy1 = Iy[x - off:x + off + 1, y - off:y + off + 1]

        Ixx = Ix1 ** 2
        Ixy = Ix1 * Iy1
        Iyy = Iy1 ** 2
        It = image2 - image1
        if It.shape != Ix1.shape:
            break
        if It.shape != Iy1.shape:
            break

        Ixt = Ix1 * It
        Iyt = Iy1 * It
        Sxx = Ixx.sum()
        Syy = Iyy.sum()
        Sxy = Ixy.sum()
        Sxt = Ixt.sum()
        Syt = Iyt.sum()
        SM = np.vstack((np.hstack((Sxx, Sxy)), np.hstack((Sxy, Syy))))
        lamb = 0.000001
        identity = np.eye(2)
        sM = SM + lamb * identity
        res = -(np.vstack((Sxt, Syt)))
        try:
            disp = np.matmul(np.linalg.inv(sM), res)
            u = disp[0]
            v = disp[1]
        except LinAlgError:
            break
        dec_u = u - int(u)
        dec_v = v - int(v)
        if abs(dec_u) > 0.5:
            if dec_u > 0:
                u = int(u) + 1
            else:
                u = int(u) - 1

        else:
            u = int(u)
        if abs(dec_v) > 0.5:
            if dec_v > 0:
                v = int(v) + 1
            else:
                v = int(v) - 1
        else:
            v = int(v)
        i = i + 1
        if U == u and V == v:
            break
        else:
            x = x + u
            y = y + v
    return (x, y)


def randomCorners(corner_list, val):
    index = []
    for i in range(val):
        num = random.randint(0, len(corner_list) - 1)
        index.append(num)
    new_corners = []
    for i in range(len(index)):
        val = index[i]
        new_corners.append(corner_list[val])

    return new_corners


def deletePoints(image, xcoor, ycoor):
    max_x = image.shape[0]
    max_y = image.shape[1]
    new_xcoor = []
    new_ycoor = []
    for i in range(len(xcoor)):
        for j in range(len(xcoor[i])):
            if xcoor[i][j] >= max_x or ycoor[i][j] >= max_y:
                new_xcoor.append(xcoor[i][j])
                new_ycoor.append(ycoor[i][j])
    return xcoor, ycoor, new_xcoor, new_ycoor


def interchange(keypoints):
    interchanged_kp = []
    for i in range(len(keypoints)):
        x = keypoints[i][0]
        y = keypoints[i][1]
        interchanged_kp.append((y, x))

    return interchanged_kp


dir = "../input/"
save_dir = "../output/"
images = readImages(dir, 51)
win = 5
k = 0.04
thresh = 500000

cornerCoor = cornerDetection(images[0], 5, 1, thresh, k, win)

list_of_key = []

try:
    choice = int(input("Choose the desired output:\n"
                       "1. Corner Detection\n"
                       "2. Feature Tracking\n\n"
                       "Your selection: "))
except ValueError:
    print("Incorrect selection. Please choose between 1 and 2")
    exit()

if choice == 1:
    plot_coor = interchange(cornerCoor)
    im = images[0].copy()
    for i in range(len(plot_coor)):
        cv2.circle(im, plot_coor[i], 2, (0, 255, 0), -1)
    cv2.imshow("Corner Detection", im)
    cv2.imwrite(save_dir + 'corners.png', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif choice == 2:
    try:
        val = int(input(
            "There are {} tracked points. Enter the percentage of points to be tracked: ".format(len(cornerCoor))))
    except ValueError:
        print("Incorrect selection. Please choose between 1 and 100")
        exit()
    if val > 0 and val <= 100:
        value = int((val / 100) * len(cornerCoor))
        cornerCoor = randomCorners(cornerCoor, value)
        list_of_key.append(cornerCoor)
        for i in range(len(images) - 1):
            img1 = images[i]
            img2 = images[i + 1]
            temp = []
            for j in range(len(list_of_key[i])):
                x = list_of_key[i][j][0]
                y = list_of_key[i][j][1]
                coor = trackPoints(x, y, img1, img2, 15)
                temp.append(coor)
            list_of_key.append(temp)

        key_array = np.asarray(list_of_key).T
        X_coor = []
        Y_coor = []
        for i in range(len(key_array[0])):
            X_coor.append(key_array[0][i])
            Y_coor.append(key_array[1][i])

        X_coor1, Y_coor1, nXcoor, nYcoor = deletePoints(images[0], X_coor, Y_coor)

        im1 = images[0].copy()
        plt.imshow(im1)
        for i in range(len(X_coor1)):
            plt.plot(Y_coor1[i], X_coor1[i], color='r')

        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_dir + 'track.png', bbox_inches='tight', pad_inches=0, format='png', dpi=300)
        plt.show()
    else:
        print("Incorrect selection. Please select between 1 and 100")

else:
    print("Incorrect selection. Please select between 1 and 2")
