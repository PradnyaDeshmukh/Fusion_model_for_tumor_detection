import numpy as np
import cv2
import os
import re
from conf import DATA_DIR
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
bins32 = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
            168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 255]) #32-bit

# bins32 = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #32-bit
# deal with labels
def normalize_label(str_): 
    str_ = str_.replace(" ","") #replace all spaces in label
    str_ = str_.translate(str_.maketrans("","","()")) #remove any ()
    str_ = str_.split("_")
    l = str_[0].split("-")
    return (l[1])


imgs = [] #list image matrix
labels = []
descs = []
dataset_dir1 = os.listdir(os.path.join(DATA_DIR,"glioma"))

def GLCM_Calculate(img_path,i,props=['dissimilarity','correlation','homogeneity','contrast','ASM','energy']):
    img = cv2.imread(img_path)
    img[:,:,0] = 0
    img[:,:,1] = 0
    # cv2.imshow('red_img',img)
    # image32 = rgb2gray(img)
    image32 = img.reshape(img.shape[0],img.shape[1]*img.shape[2])
    # cv2.imshow('gray',image32)
    np.clip(image32, 0, 255, out=image32)
    image = image32.astype('uint8')
    inds = np.digitize(image, bins32)

    # max_value = inds.max()+1
    matrix_coocurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True, symmetric=True)
    

    featture = []
    
    glcm_props = [propery for name in props for propery in graycoprops(matrix_coocurrence, name)[0]]
    
    
    for m in range(0,len(glcm_props),4):
        j = m
        avg = 0
        while(j<m+4):
            avg+=glcm_props[j]
            j+=1
        avg/=4
        featture.append(avg)


    featture.append(normalize_label(i))
    return featture


glcm_all_angls = []

for i in dataset_dir1:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"glioma"),i)
    glcm_all_angls.append(GLCM_Calculate(img_path,i))
    # print(img_path)


dataset_dir2 = os.listdir(os.path.join(DATA_DIR,"meningioma"))
for i in dataset_dir2:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"meningioma"),i)
    glcm_all_angls.append(GLCM_Calculate(img_path,i))


dataset_dir3 = os.listdir(os.path.join(DATA_DIR,"pituitary"))
for i in dataset_dir3:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"pituitary"),i)
    glcm_all_angls.append(GLCM_Calculate(img_path,i))


dataset_dir4 = os.listdir(os.path.join(DATA_DIR,"notumor"))
for i in dataset_dir4:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"notumor"),i)
    glcm_all_angls.append(GLCM_Calculate(img_path,i))

properties = ['dissimilarity','correlation','homogeneity','contrast','ASM','energy']

columns = []
angles = ['0','45','90','135']
for name in properties :
        columns.append(name)
        
columns.append("label")

import pandas as pd

glcm_df = pd.DataFrame(glcm_all_angls,columns=columns)
glcm_df.to_csv("RedtrainSet.csv")
