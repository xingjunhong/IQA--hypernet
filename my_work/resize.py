# import cv2
#
# img = cv2.imread(r'demo/2/000000270175_000004.jpg')
# img = cv2.resize(img,(128,384))
# img = cv2.imwrite(r'demo/Resize/000000270175_000004.jpg',img)
import cv2
import numpy as np
inputImage = cv2.imread(r'demo/2/000000270175_000004.jpg', 1)

outputImage = cv2.copyMakeBorder(inputImage,37,38,44,44,cv2.BORDER_CONSTANT,value=[255,255,255])

resized = cv2.resize(inputImage, (128,128), interpolation = cv2.INTER_AREA)
cv2.imwrite('output.jpg', outputImage)
cv2.imwrite('resized.jpg', resized)