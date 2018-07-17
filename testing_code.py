import numpy as np
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


cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
cellstest = [np.hsplit(row,2) for row in np.vsplit(grayimg,2)]


x = np.array(cells)
xtest = np.array(cellstest)


train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
testmine = xtest[:,:2].reshape(-1,400).astype(np.float32) # Size = (2500,400)

"""print(test)
print("---------")
print(testmine)
"""


k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()


knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE,train_labels)
ret,result,neighbours,dist = knn.findNearest(testmine,k=2)


print(result)
"""matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy"""
