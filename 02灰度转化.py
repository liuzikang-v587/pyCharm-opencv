#导入cv模块
import cv2 as cv
#读取图片
img = cv.imread('face02.jpg')
#灰度转化
gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#显示灰度
cv.imshow('gray',gray_img)
#保存灰度图片
cv.imwrite('gray_face02.jpg',gray_img)
#显示图片
cv.imshow('read_img',img)
#等待
cv.waitKey(0)
#释放内存
cv.destroyAllWindows()
