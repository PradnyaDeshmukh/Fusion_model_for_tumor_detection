import cv2
from PIL import Image
img = cv2.imread('Datasets\Training\glioma\Tr-gl_0010.jpg')
from skimage.color import rgb2gray
import numpy as np
# img = Image.open(r"Datasets\Training\glioma\Tr-gl_0010.jpg")
# img[:,:,2] = 0 #green


# img[:,:,0] = 0 # red
# img[:,:,1] = 0 # red

img[:,:,2] = 0 #blue
img[:,:,1] = 0 #blue

bins32 = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
            168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 255]) #32-bit

cv2.imshow('blue_img', img)
# cv2.waitKey()
print(img.shape)
print(img)
# image32 = img.reshape(img.shape[0],img.shape[1]*img.shape[2]).sum(1)
# np.clip(image32, 0, 255, out=img)
# image = image32.astype('uint8')
# print(image32.shape)
# print(image32)
im = np.clip(img, 0, 255, out=img)
print(img.shape)
image = img.astype('uint8')
# inds = np.digitize(image, bins32)
# image1 = inds.reshape(inds[0]*inds[1])
# print(image1.shape)
# cv2.imshow('gray',image32)
# cv2.waitKey()
# print(image32.shape)
# im1 = Image.Image.split(img)
# im1[0].show()
# im1[1].show()
# im1[2].show()
# B, G, R = cv2.split(img) 
# print(B)
# print(G)
# print(R)