#导入cv模块
import cv2 as cv

def face_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray,1.06,5,0,(10,10),(30,30))
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('ressut',img)


#读取图像
img = cv.imread('moreface01.jpg')
#检测函数
face_detect_demo()
#等待
while True:
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()
