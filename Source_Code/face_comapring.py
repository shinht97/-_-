import numpy as np
from vidgear.gears import CamGear
import cv2
from deepface import DeepFace
import mediapipe as mp

target = cv2.imread("../Images/face.jpg")

# cv2.imshow("putin", target)

url = "https://www.youtube.com/watch?v=PycdZxVbvxU"

stream = CamGear(source=url, stream_mode=True, logging=True).start()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    h, w, _ = target.shape

    results = face_detection.process(target)

    bbox = results.detections[0].location_data.relative_bounding_box

    target = target[int(h * bbox.ymin):int(h * bbox.ymin) + int(h * bbox.height),
               int(w * bbox.xmin):int(w * bbox.xmin) + int(w * bbox.width)]

    cv2.imshow("crop", target)

    while True:
        image = stream.read()
        image = cv2.resize(image, (700, 500))
        # read images

        # check if image is None
        if image is None:
            # if True break the infinite loop
            break

        # do something with image here
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                test_img = image[int(h * bbox.ymin):int(h * bbox.ymin) + int(h * bbox.height),
                       int(w * bbox.xmin):int(w * bbox.xmin) + int(w * bbox.width)]

                result = DeepFace.verify(img1_path=test_img, img2_path=target, model_name="Facenet", detector_backend="mediapipe", enforce_detection=False)

                print(result)
                # print(result["verified"])

                # if result["verified"]:
                #     print("Find Putin")

                if result["distance"] < 0.60:
                    print("Find Putin")

                mp_drawing.draw_detection(image, detection)

        cv2.imshow("Output image", image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

cv2.destroyAllWindows()

stream.stop()
