import cv2
import pickle
import numpy as np
import cvzone

# Open the input video file
cap = cv2.VideoCapture('Test.mp4')

# Load the points from the shelf file
with open("shelf", "rb") as f:
    points = pickle.load(f)

# Define the width and height of the cropped regions
w, h = 66, 189

# Define a VideoWriter object to save the output video
output_video = cv2.VideoWriter('output.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               cap.get(cv2.CAP_PROP_FPS),
                               (1020, 800))


# Define the crop function
def crop(f):
    counter = 0
    for pts in points:
        x, y = pts
        crop = f[y:y + h, x:x + w]
        count = cv2.countNonZero(crop)
        cv2.putText(frame, str(count), (x, y), 4, cv2.FONT_HERSHEY_PLAIN, (255, 0, 255), 1)
        if count > 100:
            cv2.rectangle(frame, pts, (pts[0] + w, pts[1] + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, pts, (pts[0] + w, pts[1] + h), (0, 0, 255), 2)
            counter = +1
            cvzone.putTextRect(frame, f'SpaceCount:-{counter}', (50, 60), 2, 2)


# Process each frame of the input video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020, 800))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameblur = cv2.GaussianBlur(gray, (5, 5), 1)
    framethreshold = cv2.adaptiveThreshold(frameblur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 105,
                                           9)
    framemedian = cv2.medianBlur(framethreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    framedilate = cv2.dilate(framemedian, kernel, iterations=1)
    crop(framedilate)
    cv2.imshow("FRAME", frame)
    output_video.write(frame)  # Write the frame to the output video

    if cv2.waitKey(100) & 0xFF == 27:
        break

# Release the input video and close the output video writer
cap.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
