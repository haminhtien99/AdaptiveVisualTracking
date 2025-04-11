import cv2
import numpy as np
import time



class TrackingBaseOpticalFlow:
    def __init__(self, threshold=3.):
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=3000, qualityLevel=0.1, minDistance=5, blockSize=7)
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0 = None
        self.gray = None
        self.mask = None    # show tracking line
        self.threshold = threshold

    def init(self, frame, bbox):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(frame)
        p0 = cv2.goodFeaturesToTrack(self.gray, mask=None, **self.feature_params)
        self.filter(p0, bbox)
        if self.p0.shape[0] < 2:
            return False

    def filter(self, corners, bbox):
        filtered_corners = []
        x1, y1 = bbox[:2]
        x2, y2 = x1 + bbox[2], y1 + bbox[3]
        for corner in corners:
            x, y = corner.ravel()
            if (x1 <= x <= x2) and (y1 <= y <= y2):
                filtered_corners.append(corner)
        self.p0 = np.array(filtered_corners)

    def update(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray, frame_gray, self.p0, None, **self.lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        keep_indices = self.filter_velocity(good_old, good_new)

        good_new = good_new[keep_indices]
        good_old = good_old[keep_indices]
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
        self.gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        if self.p0.shape[0] < 2:
            return False, None
        else:
            return True, self.convert_corners_to_bbox(self.p0)

    def convert_corners_to_bbox(self, corners):
        x1, y1, x2, y2 = min(corners[:, 0, 0]), min(corners[:, 0, 1]), max(corners[:, 0, 0]), max(corners[..., 1])
        return (x1.item(), y1.item(), x2.item() - x1.item(), y2.item() - y1.item())
    

    def filter_velocity(self, pts1, pts2)-> np.ndarray:
        velocities = pts2 - pts1
        mean_velocites = np.mean(velocities)
        std_velocities = np.std(velocities)
        z_scores = (velocities - mean_velocites) / std_velocities
        keep_indices = np.where(np.linalg.norm(z_scores, axis=1) <= self.threshold)
        return keep_indices

    def get_mask(self):
        return self.mask.copy()

if __name__ == '__main__':

    import os
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    # Load video
    cap = cv2.VideoCapture('/home/ha/projects/AdaptiveVisualTracking/tank2.mp4')
    # Read first frame
    ret, old_frame = cap.read()
    height, width = old_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/home/ha/projects/AdaptiveVisualTracking/output_video.mp4', fourcc, 30.0, (width, height))


    tracker = TrackingBaseOpticalFlow()

    # add cv2_track to compare
    # cv2_track = None
    cv2_track = cv2.legacy.TrackerCSRT_create()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # init_bbox= (765, 259, 95, 95) # for tank-1.mp4
    init_bbox= (370, 122, 167, 116)   # for tank-2.mp4

    tracker.init(old_frame, init_bbox)
    mask = None
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