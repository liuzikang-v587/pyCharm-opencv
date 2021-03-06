#导入cv模块
import cv2 as cv

def face_detect_demo(img):
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('ressut',img)


#读取摄像头
cap = cv.VideoCapture(0)
# 读取视频cc
# cap = cv.VideoCapture('vide01.mp4')
#循环
while True:
    flag,frame =cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()

#释放摄像头
cap.release()
