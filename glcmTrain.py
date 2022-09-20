import numpy as np
import cv2
import os
import re
from conf import DATA_DIR,TEST_DIR

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
for i in dataset_dir1:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"glioma"),i)
    # print(img_path)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray_img.shape
    print(h," ",w)
    ymin,ymax,xmin,xmax = h//3,h*2//3,w//3,w*2//3
    crop = gray_img[ymin:ymax,xmin:xmax]
    resized = cv2.resize(crop,(0,0),fx=0.5,fy=0.5)
    imgs.append(resized)
    labels.append(normalize_label(i))
    print(normalize_label(i))


dataset_dir2 = os.listdir(os.path.join(DATA_DIR,"meningioma"))
for i in dataset_dir2:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"meningioma"),i)
    # print(img_path)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray_img.shape
    print(h," ",w)
    ymin,ymax,xmin,xmax = h//3,h*2//3,w//3,w*2//3
    crop = gray_img[ymin:ymax,xmin:xmax]
    resized = cv2.resize(crop,(0,0),fx=0.5,fy=0.5)
    imgs.append(resized)
    labels.append(normalize_label(i))
    print(normalize_label(i))


dataset_dir3 = os.listdir(os.path.join(DATA_DIR,"pituitary"))
for i in dataset_dir3:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"pituitary"),i)
    # print(img_path)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray_img.shape
    print(h," ",w)
    ymin,ymax,xmin,xmax = h//3,h*2//3,w//3,w*2//3
    crop = gray_img[ymin:ymax,xmin:xmax]
    resized = cv2.resize(crop,(0,0),fx=0.5,fy=0.5)
    imgs.append(resized)
    labels.append(normalize_label(i))
    print(normalize_label(i))



dataset_dir4 = os.listdir(os.path.join(DATA_DIR,"notumor"))
for i in dataset_dir4:
    print(i)
    img_path = os.path.join(os.path.join(DATA_DIR,"notumor"),i)
    # print(img_path)
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray_img.shape
    print(h," ",w)
    ymin,ymax,xmin,xmax = h//3,h*2//3,w//3,w*2//3
    crop = gray_img[ymin:ymax,xmin:xmax]
    resized = cv2.resize(crop,(0,0),fx=0.5,fy=0.5)
    imgs.append(resized)
    labels.append(normalize_label(i))
    print(normalize_label(i))
# import imshow
# imshow(imgs[5])
print(labels)
from skimage.feature import graycomatrix, graycoprops


def rescale(img):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(img)
    scaler.transform(img)
    return img



def calculate_GLCM(img,label,props,dists=[5],angls=[0,np.pi/4,np.pi/2,2*np.pi/3],lvl=256,sym=True,norm=True):
    img = rescale(img)
    glcm = graycomatrix(img,distances=dists,angles=angls,levels=lvl,symmetric=sym,normed=norm)

    featture = []

    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        featture.append(item)

    featture.append(label)

    return featture

glcm_all_angls = []
properties = ['dissimilarity','correlation','homogeneity','contrast','ASM','energy']
for img,label in zip(imgs,labels):
    glcm_all_angls.append(calculate_GLCM(img,label,props=properties))

columns = []
angles = ['0','45','90','135']
for name in properties :
    for ang in angles:
        columns.append(name + "_" + ang)
        
columns.append("label")

import pandas as pd

glcm_df = pd.DataFrame(glcm_all_angls,columns=columns)
glcm_df.to_csv("trainSet.csv")
# print(glcm_df.head(20))


