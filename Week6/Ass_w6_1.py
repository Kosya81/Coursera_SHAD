# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 23:06:00 2016

@author: Kosya
"""

from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import img_as_float
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error


#грузим картинку
image = imread('C:\\Users\Алексей\Documents\GitHub\Coursera_SHAD\Week6\parrots.jpg')
data = img_as_float(image)

#готовим данные в матрицу
w, h, d = original_shape = tuple(data.shape)
assert d == 3
image_array = np.reshape(data, (w * h, d))
original_img = np.copy(image_array)
image_array_mean = np.copy(image_array)
image_array_median = np.copy(image_array)

#обучаем
for n_clusters in range(2,21):
    kmeans =  KMeans(n_clusters=n_clusters,init='k-means++', random_state=241).fit(image_array)
    labels = kmeans.predict(image_array)
    
    #усредняем значение в пикселях
    for cluster in range(0, n_clusters):
        for i in range (3):
            image_array_mean[labels == cluster,i] = np.mean(original_img[labels == cluster,i])
            image_array_median[labels == cluster,i] = np.median(original_img[labels == cluster,i])
    
        
    #считаем метрики
    MSE_mean = mean_squared_error(image_array,image_array_mean)
    MSE_median = mean_squared_error(image_array,image_array_median)
    
    MAXI = 1 
    PSNR_mean = 10*np.log10(MAXI**2/MSE_mean)
    PSNR_median = 10*np.log10(MAXI**2/MSE_median)
    
    print('No of clusters = %i, PSNR (mean) = %f' %(n_clusters, PSNR_mean))
    print('No of clusters = %i, PSNR (median) = %f' %(n_clusters, PSNR_median))
    
#    # Display all results, alongside original image
#    plt.figure(1)
#    plt.clf()
#    ax = plt.axes([0, 0, 1, 1])
#    plt.axis('off')
#    plt.title('Original image')
#    plt.imshow(data)
#    
#    plt.figure(2)
#    plt.clf()
#    ax = plt.axes([0, 0, 1, 1])
#    plt.axis('off')
#    plt.title('Mean')
#    plt.imshow(np.reshape(image_array_mean, original_shape))
#    
#    plt.figure(3)
#    plt.clf()
#    ax = plt.axes([0, 0, 1, 1])
#    plt.axis('off')
#    plt.title('Median')
#    plt.imshow(np.reshape(image_array_median, original_shape))
