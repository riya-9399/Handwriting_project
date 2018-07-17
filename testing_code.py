﻿import numpy as np
import cv2
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as pl

img = cv2.imread('digits.png')
testimg = cv2.imread('test.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grayimg = cv2.cvtColor(testimg,cv2.COLOR_BGR2GRAY)

cv2.imshow('image',testimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
cellstest = [np.hsplit(row,2) for row in np.vsplit(grayimg,2)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
xtest = np.array(cellstest)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
testmine = xtest[:,:2].reshape(-1,400).astype(np.float32) # Size = (2500,400)

"""print(test)
print("---------")
print(testmine)
"""

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(testmine,k=2)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
print(result)
"""matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy"""