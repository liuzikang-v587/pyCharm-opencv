# 简单图像处理opencv

## 一般操作附源码

----



### 01读取图像

```py
#导入cv模块
import cv2 as cv
#读取图片
img = cv.imread('face02.jpg')
#显示图片
cv.imshow('read_img',img)
#等待
cv.waitKey(0)
#释放内存
cv.destroyAllWindows()

```

### 

---
### 02 转化灰度图象

```py
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

```

----

### 03修改尺寸

```py
#导入cv模块
import cv2 as cv
#读取图片
img = cv.imread('face02.jpg')
#修改尺寸
resize_img = cv.resize(img,dsize=(200,200))

#显示原图片
cv.imshow('img',img)
#显示修改后图片
cv.imshow('resize_img',resize_img)
#打印原图尺寸大小
print('未修改',img.shape)
#打印修改后大小
print('未修改',resize_img.shape)

#等待
while True:
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()


```



---

### 绘制矩形

```py
#导入cv模块
import cv2 as cv
#读取图片
img = cv.imread('face02.jpg')
#坐标
x,y,w,h = 100,100,100,100
#绘制矩形
cv.rectangle(img,(x,y,x+w,y+h),color=(0,0,255),thickness=1)
#绘制圆形
cv.circle(img,center=(x+w,y+h),radius=100,color=(255,0,0),thickness=2)
#显示
cv.imshow('re_img',img)

while True:
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()

```



---

### 人脸检测

```py
#导入cv模块
import cv2 as cv

def face_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    #加载分类器
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('ressut',img)


#读取图像
img = cv.imread('face04.jpg')
#检测函数
face_detect_demo()
#等待
while True:
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()

```

CascadeClassifier，是Opencv中做人脸检测的时候的一个级联分类器。并且既可以使用Haar，也可以使用LBP特征。分类器是学习已存入的数据的规则，然后根据这些规则去检验输入的图片。

Haar特征是一种反映图像的灰度变化的，像素分模块求差值的一种特征。它分为三类：边缘特征、线性特征、中心特征和对角线特征。用黑白两种矩形框组合成特征模板，在特征模板内用 黑色矩形像素和 减去 白色矩形像素和来表示这个模版的特征值。例如：脸部的一些特征能由矩形模块差值特征简单的描述，如：眼睛要比脸颊颜色要深，鼻梁两侧比鼻梁颜色要深，嘴巴比周围颜色要深等。但矩形特征只对一些简单的图形结构，如边缘、线段较敏感，所以只能描述在特定方向（水平、垂直、对角）上有明显像素模块梯度变化的图像结构。这样就可以进行区分人脸。
————————————————
版权声明：本文为CSDN博主「位沁」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_43543515/article/details/115658773

按照自己的想法，这个函数是一个选择器，输入是xml数据形式，进行调用detectMultiScale，

detectMultiScale输入灰度变化





二、detectMultiScale函数详解




cvHaarDetectObjects是opencv1中的函数，opencv2中人脸检测使用的是 detectMultiScale函数。它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用：




void detectMultiScale(
	const Mat& image,
	CV_OUT vector<Rect>& objects,
	double scaleFactor = 1.1,
	int minNeighbors = 3, 
	int flags = 0,
	Size minSize = Size(),
	Size maxSize = Size()
);


函数介绍：

参数1：image--待检测图片，一般为灰度图像加快检测速度；

参数2：objects--被检测物体的矩形框向量组；
参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。
        如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。
        如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
        这种设定值一般用在用户自定义对检测结果的组合程序上；
参数5：flags--要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为

        CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
    
        因此这些区域通常不会是人脸所在区域；
参数6、7：minSize和maxSize用来限制得到的目标区域的范围。
————————————————
版权声明：本文为CSDN博主「walker lee」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/itismelzp/article/details/50379359



rectangle函数是用来绘制一个矩形框的，通常用在图片的标记上。

rectangle(img2, Point(j,i), Point(j + img4.cols, i + img4.rows), Scalar(255, 255, 0), 2, 8);
1
img2:要做处理的图片
二三代表左上右下矩形的角坐标
scalar：颜色
2代表线条宽度
8是线型，默认取8

Rect函数也是画矩形的，但与上面的有所不同
Rect(x,y,width,height)，x, y 为左上角坐标, width, height 则为长和宽。

Rect roi_rect = Rect(128, 128, roi.cols, roi.rows);
————————————————
版权声明：本文为CSDN博主「从刻意到习惯」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_43491924/article/details/85218336



---

### 检测多个

```py
#导入cv模块
import cv2 as cv

def face_detect_demo():
    gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    face_detect = cv.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray)
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('ressut',img)


#读取图像
img = cv.imread('face04.jpg')
#检测函数
face_detect_demo()
#等待
while True:
    if ord('q') == cv.waitKey(0):
        break

#释放内存
cv.destroyAllWindows()

```

多个人脸识别操作与人脸识别相同，只是为了更为高效识别人脸，我们需要对人脸的范围进行限定，让人脸在识别范围内

---

### 视频检测

```py
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
# cap = cv.VideoCapture(0)
# 读取视频
cap = cv.VideoCapture('vide01.mp4')
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


```

比增加了图片来源，来源于视频

从相机捕获视频
通常，我们必须使用摄像机捕获实时流。OpenCV提供了一个非常简单的界面来执行此操作。让我们从相机捕获视频（我正在使用笔记本电脑上的内置网络摄像头），将其转换为灰度视频并显示。只是一个简单的任务即可开始。

要捕获视频，您需要创建一个VideoCapture对象。它的参数可以是设备索引或视频文件的名称。设备索引仅仅是指定哪个摄像机的编号。通常，将连接一台摄像机（以我的情况为例）。所以我只是传递0（或-1）。您可以通过传递1来选择第二台摄像机，依此类推。之后，您可以逐帧捕获。但最后，不要忘记释放捕获。
————————————————
版权声明：本文为CSDN博主「瑾明达2号」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_43087913/article/details/109131611

---

### 拍照保存

```py
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
        cv2.imwrite("C:/Users/LIUZIKANG/Desktop/video_img"+str(num)+".name"+".jpg",Vshow)
        print("success to save"+str(num)+".jpg")
        print("-------------------")
        num += 1
    elif k == ord(' '):#退出
        break

#释放摄像头
cap.release()
#释放内存
cv.destroyAllWindows()
 
```



---

### 数据训练

```py
import os
import cv2
import sys
from PIL import Image
import numpy as np
def getImageAndLabels(path):
    #储存人脸数据
    faceSamples = []
    #存储姓名数据
    ids=[]
    #存储图片信息
    imgePaths=[os.path.join(path,f)for f in os.listdir(path)]
    #加载分类器
    face_detetor = cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    #遍历列表中的图片
    for imagePath in imgePaths:
        #打开图片，灰度化，PIL有九种不同模式：1，L,P,RGB,RGBA,CMYK.YCbCr,I，F
        PIL_img = Image.open(imagePath).convert('L')
        #将图像转换为数组，以黑白深浅
        img_numpy=np.array(PIL_img,'uint8')
        #获取图片人脸特征
        faces = face_detetor.detectMultiScale(img_numpy)
        #获取每张图片的id和姓名
        id = int(os.path.split(imagePath)[1].split('.')[0])
        #预防无面容照片
        for x,y,w,h in faces:
            ids.append(id)
            faceSamples.append(img_numpy[y:y+h,x:x+w])
        #打印脸部特征和id
    print('id',id)
    print('fs:',faceSamples)
    return faceSamples,ids




if __name__ == '__main__':
    #图片路径
    path='./data/jm/'
    #获取图像数组和id标签数组和姓名
    faces,ids = getImageAndLabels(path)
    #加载识别器
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    #训练
    recognizer.train(faces,np.array(ids))
    #保存文件
    recognizer.write('trainer/trainer.yml')


```



---

### 人脸识别

```py
import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib

#加载训练数据集文件
recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainer.yml')
names=[]
warningtime = 0

#准备识别的图片
def face_detect_demo(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    face_detector=cv2.CascadeClassifier('D:/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    #级联分类器
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    #face=face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        if confidence > 80:#识别阈值当大于此时，会发出警告
            global warningtime
            warningtime += 1
            if warningtime > 100:
               warningtime = 0
            cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result',img)
    #print('bug:',ids)

def name():
    path = './data/jm/'#打开训练好的数据库
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)


cap=cv2.VideoCapture(0)#打开电脑摄像头
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):#按空格退出摄像头
        break
cv2.destroyAllWindows()#关闭，释放相应的内存
cap.release()
#print(names)


```







