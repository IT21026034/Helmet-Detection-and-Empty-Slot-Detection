import cv2
import pickle

cap = cv2.VideoCapture('Test.mp4')

try:
    with open("shelf", "rb") as f:
        points = pickle.load(f)
except:
    points = []


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
    print(point)


def drawrect(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        with open("shelf", "wb") as f:
            pickle.dump(points, f)
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, pts in enumerate(points):
            x1, y1 = pts
            points.pop(i)


# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)
w, h = 66, 189
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame, (1020, 800))
    for pts in points:
        cv2.rectangle(frame, pts, (pts[0] + w, pts[1] + h), (0, 0, 255), 2)
    # cv2.rectangle(frame, (432, 195), (498, 384), (0, 0, 255), 2)

    cv2.imshow("FRAME", frame)
    cv2.setMouseCallback('FRAME', drawrect)
    if cv2.waitKey(100) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
