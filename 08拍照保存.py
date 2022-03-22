#导入cv模块
import cv2
import cv2 as cv

#读取摄像头
cap = cv.VideoCapture(0)

flag = 1
num = 1
while(cap.isOpened()):#检测是否在开启状态
    ret_flag,Vshow = cap.read()#得到每一帧视频
    cv2.imshow('Capture_Test',Vshow)#显示图像
    k = cv2.waitKey(1) & 0XFF#按键判断
    if k == ord('s'):
        cv2.imwrite("data/jm/"+str(num)+".liuzikang"+".jpg",Vshow)
        print("success to save"+str(num)+".jpg")
        print("-------------------")
        num += 1
    elif k == ord(' '):#退出
        break

#释放摄像头
cap.release()
#释放内存
cv.destroyAllWindows()
 