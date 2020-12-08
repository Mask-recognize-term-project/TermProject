# [ 실햄법 ] 아래의 코드를 Terminal에 입력.
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import os

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2


from tensorflow.python.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# 모델 불러오기
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# 카메라 시작
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

model = load_model("mask_prediction")

data_list = []
loc = []

# 프레임을 계속 추출
while True:
	# 캠에서 프레임별 사진 추출 후 크기 조정
	frame = vs.read()
	frame = imutils.resize(frame, width=400)


	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
								 (300, 300), (104.0, 177.0, 123.0))


	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]


		# confidence의 최솟값보다 큰 것만 걸러냄
		if confidence < args["confidence"]:
			continue

		# 인식된 물체의 테두리 좌표 계산
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		face = frame[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (64, 64))
		face = np.expand_dims(face, axis=0)
		face = np.asarray(face)
		data_list.append(face)
		loc.append((startX, startY, endX, endY))

		if len(data_list) > 0:
			predictions = model.predict(face)
			print(predictions[0])
			print(np.argmax(predictions[0]))
		for box in loc:
			(startX, startY, endX, endY) = box
			if np.argmax(predictions[0]) == 0:
				text = "Mask"
				color = (0,255,0)
			elif np.argmax(predictions[0]) == 1:
				text = "Mask_Mouth_Chin"
				color = (0, 140, 255)
			elif np.argmax(predictions[0]) == 2:
				text = "Mask_Chin"
				color = (0, 0, 255)
			else :
				text = "No Mask"
				color = (255,0,0)

		text = "{}:".format(text)
		cv2.rectangle(frame, (startX, startY), (endX, endY),
					  color, 2)
		cv2.putText(frame, text, (startX, startY-10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q'버튼 누르면 종료
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()