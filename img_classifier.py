import cv2
import numpy as np
import os
 
path = 'C:/Users/SAJJAD IJAZ/Desktop/ImagesQuery'
images = []
classNames= []
mylist = os.listdir(path)
print(mylist)
print('Total Classes Detected', len(mylist))

for cl in mylist:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findDes(images):
    deslist=[]
    for img in images:
        orb = cv2.ORB_create()
        kp,des = orb.detectAndCompute(img,None)
        deslist.append(des)
    return deslist

def findID(img, desList,thres=15):
    orb = cv2.ORB_create()
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList=[]
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList)!=0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal

deslist=findDes(images)
print(len(deslist))

cap=cv2.VideoCapture(0)
while True:
    ret,img2=cap.read()
    if img2 is None:
        break
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ide=findID(imgOriginal,deslist)
    if ide != -1:
        cv2.putText(imgOriginal,classNames[ide],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv2.imshow("Camera",imgOriginal)
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()
