import cv2
from PIL import Image
img = cv2.imread('Datasets\Training\glioma\Tr-gl_0010.jpg')
from skimage.color import rgb2gray

# img = Image.open(r"Datasets\Training\glioma\Tr-gl_0010.jpg")
# img[:,:,2] = 0 #green


# img[:,:,0] = 0 # red
# img[:,:,1] = 0 # red

img[:,:,2] = 0 #blue
img[:,:,1] = 0 #blue


cv2.imshow('red_img', img)
cv2.waitKey()
print(img.shape)

image32 = rgb2gray(img)
cv2.imshow('gray',image32)
cv2.waitKey()
print(image32.shape)
# im1 = Image.Image.split(img)
# im1[0].show()
# im1[1].show()
# im1[2].show()
# B, G, R = cv2.split(img) 
# print(B)
# print(G)
# print(R)