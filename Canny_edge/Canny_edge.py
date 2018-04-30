from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter
import numpy as np
from skimage.io import imread
from skimage.io import imshow
import matplotlib.pyplot as plt

def crop(start, end, origin_image):
    return origin_image[start[0]:end[0]+1,start[1]:end[1]+1] 

def sobel_filter(img, x= True):
    if x:
        filter_ =[[-1 ,0, 1],
              [-2 ,0, 2],
              [-1 ,0, 1]]
    else:
        filter_ =[[-1 ,-2, -1],
               [0 , 0, 0],
               [1 , 2, 1]]
    new_img = np.zeros(img.shape)
    X, Y = new_img.shape
    for i in range(X):
        for j in range(Y):
             start = (i-1,j-1)
             end =  (i+1,j+1)
             if i-1<0 or j-1<0 or i+1>X-1 or j+1 >Y-1:
                 continue 
             crop_img=crop(start, end, img)
             for ii in range(3):
                 for jj in range(3):
                     new_img[i,j] +=crop_img[ii,jj]*filter_[ii][jj] 
    return new_img


def angle(a):
        a = np.rad2deg(a) % 180
        if (0 <= a < 22.5) or (157.5 <= a < 180):
            a = 0
        elif (22.5 <= a < 67.5):
           a = 45
        elif (67.5 <= a < 112.5):
            a = 90
        elif (112.5 <= a < 157.5):
            a = 135
        return a

def gradient(img):
    sobelx = sobel_filter(image)
    sobely = sobel_filter(image, False)
    img_hypot = np.hypot(sobelx, sobely)     #(sobelx[i,j]**2+sobely[i,j]**2)**0.5
    img_dir = np.arctan2(sobely,sobelx)
    img_hypot = img_hypot/img_hypot.max()
    return img_hypot,img_dir

def suppression(img, img_direction):
    X, Y = img.shape
    img_suppr = np.zeros((X,Y))
    for i in range(X):
        for j in range(Y):
            where = angle(img_direction[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        img_suppr[i,j] = img[i,j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                       img_suppr[i,j] = img[i,j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        img_suppr[i,j] = img[i,j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        img_suppr[i,j] = img[i,j]
            except IndexError as e:
               pass
    return img_suppr

def threshold(img,lower,upper):
    a = { 'W': 50,'S': 255}
    strong_i, strong_j = np.where(img > upper)
    weak_i, weak_j = np.where((img >= lower) & (img <= upper))
    zero_i, zero_j = np.where(img < lower)
    img[strong_i, strong_j] = a.get('S')
    img[weak_i, weak_j] = a.get('W')
    img[zero_i, zero_j] = 0
    return img 


def edge_track(img):
    weak = 50
    strong = 255
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                         or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                         or (img[i+1, j + 1] == strong) or (img[i-1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img





image = imread("D:\\Desktop - HDD\\Images\\lena.jpg")
img_gauss = gaussian_filter(image,8)
img_hypo, img_direc = gradient(img_gauss)

img_suppression = suppression(img_hypo, img_direc)

mean = np.mean(img_gauss)
mn = .25 * mean
upper = mean + mn
lower = mean - mn
img_thresh = threshold(img_gauss, lower, upper)

img_tracking = edge_track(img_thresh)



fig = plt.figure(figsize=(24,24))
fig.add_subplot(2,2,1)
imshow(image)
fig.add_subplot(2,2,2)
imshow(img_hypo)
fig.add_subplot(2,2,3)
imshow(img_suppression)
print(img_hypo.size)
print(img_suppression.size)
fig.add_subplot(2,2,4)
imshow(img_thresh)
#fig.add_subplot(3,2,5)
#imshow(img_tracking)
plt.show()
