import cv2 as cv
import sys

img=cv.imread('source/ch2/soccer.jpg')
# 이미지 파일을 읽어와서 NumPy배열 형식으로 반환하는 함수
# OpenCV에서 이미지 작업을 시작할 때 가장 먼저 사용하는 필수 함수


if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('Image Display', img)

cv.waitKey()
cv.destroyAllWindows()