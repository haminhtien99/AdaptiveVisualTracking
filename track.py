import cv2
import os
import time
from matchers.sparseOptFlow import TrackingBaseOpticalFlow

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Load video
video_and_bbox = {1: ['tank1.mp4', (765, 259, 95, 95)],
                  2: ['tank2.mp4', (370, 122, 167, 130)],
                  3: ['tank3.mp4', (284, 92, 228, 124)],
                  4: ['tank4.mp4', (140, 196, 355, 170)]}
video, init_bbox = video_and_bbox[1]

cap = cv2.VideoCapture(video)

# Read first frame
ret, old_frame = cap.read()
height, width = old_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (width, height))


tracker = TrackingBaseOpticalFlow(dist_similarity=-0.6, retention_ratio=0.8)

# add cv2_track to compare
cv2_track = None
# cv2_track = cv2.legacy.TrackerCSRT_create()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


tracker.init(old_frame, init_bbox)
mask = None # show tracklets of features object
if cv2_track is not None:
    cv2_track.init(old_frame, init_bbox)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    success, bbox = tracker.update(frame)

    if cv2_track is not None:
        success_1, bbox_1 = cv2_track.update(frame)

    
    if success:
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        pt1 = (x,y)
        pt2 = (x + w, y + h)
        cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
        if cv2_track is not None:
            if success_1:
                x, y, w, h = bbox_1
                x, y, w, h = int(x), int(y), int(w), int(h)
                pt1 = (x,y)
                pt2 = (x + w, y + h)
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
        for point in tracker.get_points():
            cv2.circle(frame, point, radius=2, color=(0, 255, 0), thickness=-1)
        # mask = tracker.get_mask()
    if mask is not None:
        img = cv2.add(frame, mask)
    else: img = frame.copy()
    track_time = time.time() - start
    fps = 1/track_time
    cv2.putText(img, f'FPS:{fps:.0f}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0))
    cv2.imshow('frame', img)
    out.write(img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()