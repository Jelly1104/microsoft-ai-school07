import cv2 as cv
import sys
img=cv.imread('source/ch2/girl_laughing.jpg')


if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.rectangle(img,(830,30),(1000,200),(0,0,255),2) # 직사각형 그리기(img, (왼쪽 위 모서리 좌표),(오른쪽 위 모서리 좌표),(B,G,R),굵기)
cv.putText(img,'Laugh',(830,24),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,0,0),2) #글씨 쓰기(img,'글씨',(좌표),폰트,글씨크키,글씨색(B,G,R),굵기)

cv.imshow('Draw',img)
cv.imwrite('Draw image.jpg',img) #사진을 저장하기



cv.waitKey()
cv.destroyAllWindows()