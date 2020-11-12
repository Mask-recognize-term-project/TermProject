import numpy as np
import cv2 as cv
import dlib

faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 얼굴의 각 구역의 포인트 구분
JAWLINE_POINTS = list(range(0,17))
RIGHT_EYEBROW_POINTS = list(range(17,22))
LEFT_EYEBROW_POINTS = list(range(22,27))
NOSE_POINTS = list(range(27,36))
RIGHT_EYE_POINTS = list(range(36,42))
LEFT_EYE_POINTS = list(range(42,48))
MOUTH_OUTLINE_POINTS = list(range(48,61))
MOUTH_INNER_POINTS = list(range(61,68))

"""
    def = dlib을 이용하여 얼굴과 눈을 찾는 함수
    input = 그레이 스케일 이미지
    output = 얼굴 중요 68개의 포인트에 그려진 점 + 이미지
"""

def detect(gray, frame):
    # 등록한 Cascade classifier 를 이용하여 얼굴 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05,
                                         minNeighbors=5, minSize=(100,100), flags=cv.CASCADE_SCALE_IMAGE)
    # 얼굴에서 랜드마크 찾기
    for (x,y,w,h) in faces:
        dlib_rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        landmarks = np.matrix([[p.x,p.y] for p in predictor(frame,dlib_rect).parts()])
        landmarks_display = landmarks[NOSE_POINTS]

        for idx, point in enumerate(landmarks_display):
            pos = (point[0,0],point[0,1])
            cv.circle(frame, pos, 2, color=(0,255,255),thickness=-1)

    return frame

# 웹캠에서 이미지 가져오기
video_capture = cv.VideoCapture(0)

while True:
    _, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    canvas = detect(gray,frame)

    cv.imshow("img",canvas)

    if cv.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv.destroyAllWindows()
