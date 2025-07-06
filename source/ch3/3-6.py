import cv2 as cv
import matplotlib.pyplot as plt

# 히스토그램 평활화

img=cv.imread('source/ch3/mistyroad.jpg') 

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)		# 명암 영상으로 변환하고 출력
plt.imshow(gray,cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h=cv.calcHist([gray],[0],None,[256],[0,256])	# 히스토그램을 구해 출력
plt.plot(h,color='r',linewidth=1), plt.show()

equal=cv.equalizeHist(gray)			# 히스토그램을 평활화하고 출력
plt.imshow(equal,cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h=cv.calcHist([equal],[0],None,[256],[0,256])	# 히스토그램을 구해 출력
plt.plot(h,color='r',linewidth=1), plt.show() 


# matplotlib에서 'q'로 닫히는게 기본 동작이 맞음.
    #cv.waitkey()가 없어도 'q'로 닫힘 ('cmd+w')
