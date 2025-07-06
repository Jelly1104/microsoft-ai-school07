#2-9 좀더 부드러운 line으로 이용해 보기

import cv2 as cv
import sys

#페인팅 

img=cv.imread('source/ch2/soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')


BrushSiz=5 #붓의 크기
Lcolor,Rcolor=(255,0,0),(0,0,255) #파란색과 빨간색

def painting(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,Lcolor,-1) #마우스 왼쪽 버튼 클릭하면 파란색
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,Rcolor,-1) #마우스 오른쪽 버튼 클릭하면 빨간색
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        if prev_pt is not None:
            cv.line(img,prev_pt,(x.y),BrushSiz*2,Lcolor) #왼쪽 버튼 클릭하고 이동하면 파란색
        prev_pt = (x,y)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        if prev_pt is not None:
            cv.line(img,prev_pt,(x.y),BrushSiz*2,Rcolor) #오른쪽 버튼 클릭하고 이동하면 빨간색
        prev_pt = (x,y)
        
    cv.imshow('painting',img)

cv.namedWindow('painting')
cv.imshow('painting',img)

cv.setMouseCallback('painting',painting)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
