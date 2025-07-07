# 📂 OpenCV를 활용한 얼굴 인식 및 이미지 처리 실습
## 📝 학습 목표

이번 학습에서는 순수 Python 라이브러리인 OpenCV와 Gradio를 사용하여, 로컬 환경에서 실행되는 실시간 영상 처리 웹 애플리케이션을 구축하는 것을 목표로 합니다. 이를 통해 실용적인 컴퓨터 비전 애플리케이션 개발의 기초부터 구조화 방법까지 다룹니다.

본 과정에서는 다음과 같은 핵심 기술들을 학습합니다.

로컬 모델 활용: Azure와 같은 클라우드 서비스 없이, 사전 훈련된 Haar Cascade XML 파일과 YOLOv3 모델(.weights, .cfg)을 직접 로드하여 사용하는 방법을 익힙니다.

실시간 영상 처리 및 UI 구축: Gradio의 스트리밍 기능을 활용하여 웹캠 영상을 실시간으로 받아오고, 매 프레임마다 OpenCV를 통해 얼굴 및 객체 탐지를 수행합니다.

결과 시각화: 탐지된 객체의 위치에 OpenCV와 Pillow(PIL)를 사용하여 바운딩 박스와 텍스트 레이블을 그리고, 처리된 영상을 사용자에게 다시 스트리밍하는 방법을 실습합니다.

## 🖼️ 프로젝트 개요
이 프로젝트는 두 가지 주요 기능으로 구성됩니다.

1. ***실시간 얼굴 인식***: 웹캠을 통해 입력받은 실시간 영상에서 얼굴을 검출하고 결과를 실시간으로 표시합니다.

2. ***이미지 처리 시각화***: OpenCV를 이용하여 얼굴 주변에 사각형을 그리고, 이미지 위에 텍스트를 추가하여 인식 결과를 명확하게 전달합니다.

이러한 기능을 통해 컴퓨터 비전 기초 지식을 활용하여 실제적인 응용 프로그램을 개발하는 방법을 익힙니다.

## 📁 파일 구성 및 설명
**파일 설명:**

| 파일명                | 설명                                                        |
| :-------------------- | :---------------------------------------------------------- |
| `src/`                | 프로젝트 소스 코드를 포함하는 디렉토리                     |
| `250707_opencv.ipynb` | OpenCV를 활용한 얼굴 인식 및 이미지 처리 실습 Jupyter Notebook |
| `README.md`           | 프로젝트에 대한 설명 문서 (현재 보고 계신 파일)           |
| `mise.toml`           | 프로젝트 설정 및 의존성 관리를 위한 파일 (예: mise, poetry 등) |
| `requirements.txt`    | 프로젝트 실행에 필요한 Python 라이브러리 목록              |



## 🚀 주요 코드 및 실행 결과
OpenCV를 이용한 실시간 얼굴 인식 (opencv_face_detection.ipynb)

본 실습에서는 OpenCV의 Haar cascade를 이용하여 웹캠으로 입력된 실시간 영상에서 얼굴을 검출합니다.

핵심 코드 (로컬 모델 활용 및 얼굴 표기)
```
import cv2
import gradio as gr

cascade_files = [  
    "haarcascade_frontalface_default.xml",
    "haarcascade_upperbody.xml",  
    "haarcascade_eye_tree_eyeglasses.xml",  
    "haarcascade_eye.xml",  
    "haarcascade_frontalcatface_extended.xml",  
    "haarcascade_frontalcatface.xml",  
    "haarcascade_frontalface_alt_tree.xml",  
    "haarcascade_frontalface_alt.xml",  
    "haarcascade_frontalface_alt2.xml",    
    "haarcascade_fullbody.xml",  
    "haarcascade_lefteye_2splits.xml",  
    "haarcascade_license_plate_rus_16stages.xml",  
    "haarcascade_lowerbody.xml",  
    "haarcascade_profileface.xml",  
    "haarcascade_righteye_2splits.xml",  
    "haarcascade_russian_plate_number.xml",  
    "haarcascade_smile.xml"  
]  


cascade_path="{}haarcascade_frontalface_default.xml".format(cv2.data.haarcascades)
print(cascade_path)
face_cascade=cv2.CascadeClassifier(cascade_path) #분류기 초기화

def detect_face(origin_image,scale_factor_number,min_neighbors_number,min_size_number):
    image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR) # CV에서는 기본적으로 BGR을 사용함.

    face_list=face_cascade.detectMultiScale(
        image=image,
        scaleFactor=scale_factor_number, #10%씩 줄여가며 이미지를 찾겠다.
        minNeighbors=min_neighbors_number, #인접한 위치에서 5개 이상 잡혀야 얼굴로 인식하겠다.
        minSize=(min_size_number,min_size_number) #얼굴 감지의 최소한의 픽셀
    )
    print(face_list)

    for face in face_list:
        x, y, w, h = face
        cv2.rectangle(image, (x, y),(x+w,y+h),(0,255,0),2) # BGR로 인식
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # ✅ BGR을 다시 RGB로 (디코드)

    return image
```
핵심 코드 (실시간 이미지 처리 시각화)
```
with gr.Blocks() as demo:

    DEFAULT_SCALE_FACTOR = 1.1
    DEFAULT_MIN_NEIGHBORS = 5
    DEFAULT_MIN_SIZE = 30 

    scale_factor = gr.State(DEFAULT_SCALE_FACTOR) 
    min_neighbors = gr.State(DEFAULT_MIN_NEIGHBORS)
    min_size = gr.State(DEFAULT_MIN_SIZE)


    def stream_webcam(image,scale_factor_number,min_neighbors_number,min_size_number):
        detected_image=detect_face(image,scale_factor_number,min_neighbors_number,min_size_number)
        return detected_image
    
    def change_haar(haar_name):
        global face_cascade 
        print(haar_name)
        face_cascade = cv2.CascadeClassifier("{}{}".format(cv2.data.haarcascades,haar_name))

    def change_scale_factor(scale_factor_text):
        return float(scale_factor_text)
    
    def click_apply(scale_factor_text,min_neighbors_text,min_size_text):
        print(scale_factor_text,min_neighbors_text,min_size_text)
        return float(scale_factor_text), int(min_neighbors_text), int(min_size_text)
        
    with gr.Column():
        with gr.Row():
            scale_factor_textbox=gr.Textbox(label="scalefactor",value=DEFAULT_SCALE_FACTOR,interactive=True)
            min_neighbors_textbox=gr.Textbox(label="minNeighbors",value=DEFAULT_MIN_NEIGHBORS,interactive=True)
            min_size_textbox=gr.Textbox(label="minSize",value=DEFAULT_MIN_SIZE,interactive=True)
        apply_button = gr.Button("적용")    

    haar_dropdown=gr.Dropdown(label="Haar CasCade 선택",choices=cascade_files, value=cascade_files[0],interactive=True)
    webcam_image=gr.Image(label="카메라",sources="webcam",streaming=True,width=480,height=270,mirror_webcam=False)
    output_image=gr.Image(label="검출화면",streaming=True,interactive=False)

    webcam_image.stream(stream_webcam, inputs=[webcam_image,scale_factor,min_neighbors,min_size],outputs=[output_image])
    haar_dropdown.change(change_haar,inputs=[haar_dropdown],outputs=[])

    scale_factor_textbox.change(change_scale_factor,inputs=[scale_factor_textbox],outputs=[scale_factor])
    apply_button.click(click_apply, inputs=[scale_factor_textbox,min_neighbors_textbox,min_size_textbox], outputs=[scale_factor,min_neighbors,min_size])
demo.launch()
```

## ✨ 실행 결과

웹캠을 통해 실시간으로 얼굴이 검출되고, 아래와 같이 사각형과 텍스트로 표시됩니다.

<table>
    <tr>
        <th>👤 haarcascade_fullbody.xml</th>
        <th>🧑‍💻 haarcascade_frontalface_default.xml</th>
    </tr>
    <tr>
        <td><img src="image_c2661b.jpg" alt="전신 감지"></td>
        <td><img src="image_c265bc.jpg" alt="얼굴 감지"></td>
    </tr>
</table>


## 💡 학습 정리

이번 세션을 통해 OpenCV를 활용하여 실시간 얼굴 인식을 구현하는 방법을 배웠습니다. Haar cascade와 같은 사전 훈련된 모델을 쉽게 활용하여 복잡한 얼굴 인식 작업을 빠르게 처리할 수 있었습니다. 이 과정에서 이미지 처리 기술을 활용하여 인식된 결과를 시각적으로 명확히 전달하는 방법도 익혔습니다. 이러한 기술은 향후 다양한 컴퓨터 비전 응용 프로그램을 구축하는 데 유용하게 사용될 것입니다.



###  🙆🏻‍♀️ About Me

Eunah Jeong (정은아)

[![GitHub](https://badgen.net/badge/icon/github%20Eunah?icon=github&label)](https://github.com/Jelly1104/microsoft-ai-school07) [![LinkedIn Badge](http://img.shields.io/badge/-LinkedIn-0072b1?style=flat&logo=linkedin&link=https://www.linkedin.com/in/eunah-jeong-02115b24b/)](https://www.linkedin.com/in/eunah-jeong-02115b24b/)

### 💌 Contact
[![Gmail Badge](https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white)](sina911104@gmail.com)
