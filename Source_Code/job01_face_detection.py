from vidgear.gears import CamGear
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
stream = CamGear(source='https://www.youtube.com/watch?v=VUOLMttofws', stream_mode=True,
                 logging=True).start()  # YouTube Video URL as input

# infinite loop
while True:

    image = stream.read()
    image = cv2.resize(image, (1920, 1080))
    # read frames

    # check if frame is None
    if image is None:
        # if True break the infinite loop
        break

    # do something with frame here

    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 성능을 향상시키려면 이미지를 작성 여부를 False으로 설정하세요.
        image.flags.writeable = False
        results = face_detection.process(image)

        # 영상에 얼굴 감지 주석 그리기 기본값 : True.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow('test', image)
    # Show output window

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        # if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

# safely close video stream.
stream.stop()