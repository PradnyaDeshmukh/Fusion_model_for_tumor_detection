from scipy import ndimage as nd
import cv2
import matplotlib.pyplot as plt



img = cv2.imread('Datasets\Training\glioma\Tr-gl_0010.jpg')

median_img = nd.median_filter(img, size=3)
edges = cv2.Canny(img, 100,200)
gaussian_img = nd.gaussian_filter(img, sigma=3)

# median_img1 = median_img.reshape(-1)

# plt.imshow(median_img)
plt.imshow(edges)
# plt.imshow(gaussian_img)
plt.show()