from vidgear.gears import CamGear
import cv2
import mediapipe as mp

url = "https://www.youtube.com/watch?v=nvJeJSrghOI"

stream = CamGear(source=url, stream_mode=True, logging=True).start()  # YouTube Video URL as input

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        image = stream.read()
        image = cv2.resize(image, (700, 500))
        # read images

        # check if image is None
        if image is None:
            # if True break the infinite loop
            break

        # do something with image here
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        cv2.imshow("Output image", image)
        # Show output window

        key = cv2.waitKey(1) & 0xFF
        # check for 'q' key-press
        if key == ord("q"):
            # if 'q' key-pressed break out
            break

cv2.destroyAllWindows()

stream.stop()