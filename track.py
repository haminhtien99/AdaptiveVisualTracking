import cv2
import os
import time
from adaptive_tracker import MyTracker

trackers_dict = {
    "csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create, 
    "custom": MyTracker
	}

Parameters = {
    'WC_threshold': 0.51,   # Threshold score for a weak confidence candidate decision
    'TD_threshold': 0.5,   # Tolerable distance between the motion- and model-based trackers
    'CA_threshold': 0.63,   # Threshold for a candidate area decision
    'NP_threshold': 0.70,   # Threshold for a negative patch determination
    'PP_threshold': 0.38,   # Threshold for a positive patch determination
    'SC_threshold': 0.61    # Threshold score for a strong confidence candidate decision
}


def show_position(bbox_roi, frame):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    cv2.line(frame, (width // 2, 0), (width // 2, height), green, 2)
    cv2.line(frame, (0, height // 2), (width, height // 2), green, 2)

    if bbox_roi is not None:
        x, y, w, h = map(int, bbox_roi)
        xc, yc = x + w//2, y + h//2
        cv2.putText(frame, f"({xc - center_x}, {center_y - yc})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue, 2)
        cv2.circle(frame, (xc, yc), 5, red, -1)
        cv2.arrowedLine(frame, (center_x, center_y), (xc, yc), red, 2, tipLength=0.1)


tracker = trackers_dict['custom'](Parameters)
# tracker = trackers_dict['csrt']()
cap = cv2.VideoCapture('tank1.mp4')
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()
bbox = cv2.selectROI("Select Object", frame, False)
tracker.init(frame, bbox)
while True:
    tick = cv2.getTickCount()
    ret, frame = cap.read()
    if not ret:
        break

    success, bbox = tracker.update(frame)
    time_taken = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
    fps = 1 / time_taken
    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        show_position(bbox, frame)
    else:
        cv2.putText(frame, "Tracking Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        show_position(None, frame)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()