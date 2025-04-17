import cv2
import numpy as np
from typing import List, Tuple

class TrackingBaseOpticalFlow:
    """
    TrackingBaseOpticalFlow implements a feature-based object tracking system
    using optical flow and descriptor-based matching.

    Core functionality:
    - Initializes feature points within a bounding box using Shi-Tomasi corner detection.
    - Tracks points using Lucas-Kanade pyramidal optical flow across frames.
    - Filters tracked points based on motion direction similarity to reduce noise.
    - Maintains tracking quality with a retention ratio threshold.
    - Reactivates lost tracking using ORB descriptors and homography estimation.
    - Dynamically adjusts and updates bounding box based on tracked points.
    - Provides visual tracking mask and feature point access.

    Attributes:
        feature_params (dict): Parameters for corner detection.
        lk_params (dict): Parameters for optical flow tracking.
        main_feat_pts (np.ndarray): Primary feature points being tracked.
        prob_feat_pts (np.ndarray): Probable new features to be added.
        retention_ratio (float): Minimum percentage of original features required to avoid reactivation.
        dist_similarity (float): Cosine similarity threshold used to filter out inconsistent motion vectors;
                                 higher values retain only more coherent motion patterns.
        last_good_crop (np.ndarray): Grayscale image crop from the last successfully tracked bounding box;
                                     used as a reference for reactivation via ORB matching.
        descriptor (cv2.ORB): ORB descriptor used for feature extraction and matching.
        matcher (cv2.BFMatcher): Brute-force matcher configured for ORB feature comparison.
    """

    def __init__(self, dist_similarity=-0.5, retention_ratio=0.7):
        """
        Initializes the tracker with specified similarity and retention thresholds.

        Args:
            dist_similarity (float): Cosine similarity threshold for motion vector filtering.
            retention_ratio (float): Threshold for percentage of retained features before reactivation.
        """
        # Parameters for Shi-Tomasi corner detection
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.1, minDistance=5, blockSize=7)
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.main_feat_pts = None
        self.prob_feat_pts = None
        self.num_init_pts = 0
        self.gray = None
        self.mask = None    # show tracking line
        self.dist_similarity = dist_similarity

        self.retention_ratio = retention_ratio
        self.last_good_crop = None

        self.descriptor = cv2.ORB_create(nfeatures=300)    #descriptor
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    def init(self, frame, bbox: Tuple[int|float]) -> bool:
        """
        Initializes the tracker with the first frame and a bounding box.

        Args:
            frame (np.ndarray): First video frame (BGR image).
            bbox (tuple): Bounding box (x, y, w, h) to initialize feature extraction.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(frame)

        self.main_feat_pts = self._extrac_feat_pts(bbox)
        if self.main_feat_pts is not None:
            self.num_init_pts=len(self.main_feat_pts)
            return True
        if self.num_init_pts < 2:
            return False


    def update(self, frame: np.ndarray):
        """
        Updates the tracker with a new frame and maintains the current state.

        Args:
            frame (np.ndarray): New video frame (BGR image).

        Returns:
            tuple: (bool, bbox) — success flag and updated bounding box.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.predict_OpticalFlow(frame_gray)
        self.gray = frame_gray.copy()   # update self.gray to next update

        if len(self.main_feat_pts) >= self.num_init_pts * self.retention_ratio:
            x, y, w, h = map(int, self._get_bbox_from_pts())
            self.last_good_crop = frame_gray[y:y+h, x:x+w]

        if len(self.main_feat_pts) / self.num_init_pts < self.retention_ratio:
            self.reactivate()
        bbox = self._get_bbox_from_pts()
        return True, bbox

    def predict_OpticalFlow(self, frame_gray: np.ndarray):
        """
        Tracks feature points from the previous frame to the current using optical flow.
        Applies velocity-based filtering and merges new candidate points.

        Args:
            frame_gray (np.ndarray): Current frame in grayscale.

        Returns:
            tuple: Updated bounding box estimated from good feature points.
        """
        # Calculate optical flow for self.main_feat_pts
        pred_main_feat, st_main, _ = cv2.calcOpticalFlowPyrLK(self.gray, frame_gray, self.main_feat_pts, None, **self.lk_params)
        # Select good points
        good_new = pred_main_feat[st_main == 1]
        good_old = self.main_feat_pts[st_main == 1]

        # Calculate optical flow for
        if self.prob_feat_pts is not None:
            pred_prob_feat, st_prob, _ = cv2.calcOpticalFlowPyrLK(self.gray, frame_gray, self.prob_feat_pts, None, **self.lk_params)
            good_new_prob = pred_prob_feat[st_prob == 1]
            good_old_prob = self.prob_feat_pts[st_prob == 1]
        else:
            good_new_prob = None
            good_old_prob = None

        first_keep, second_keep = self._filter_velocity((good_old, good_old_prob), (good_new, good_new_prob))

        good_new = good_new[first_keep]
        good_old = good_old[first_keep]

        if second_keep is not None:
            good_new_prob = good_new_prob[second_keep]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)

        # add self.prob_feat_pts to self.main_feat_pts
        if good_new_prob is not None:
            self.main_feat_pts = np.concatenate((good_new, good_new_prob), axis=0).reshape(-1, 1, 2)
            self.prob_feat_pts = None
        else:
            self.main_feat_pts = good_new.reshape(-1, 1, 2)

    def reactivate(self):
        """
        Attempts to recover tracking when the number of good feature points falls below the threshold.
        Uses ORB-based matching and homography to reinitialize the tracking region.
        """
        # Use the last known good bbox or fallback
        use_bbox = self._get_bbox_from_pts()

        # Enlarge the box more generously
        x_new, y_new, w_new, h_new = self._upscale(use_bbox, scale=1.5)
        crop = self.gray[y_new: y_new + h_new, x_new: x_new + w_new]

        # features matching with orb descriptor and matcher
        corners = self._matching(crop)
        if corners is not None:
            matched_bbox = corners + np.tile((x_new, y_new), (4,1)).astype(np.float32)
            x_min = min(matched_bbox[:, 0])
            y_min = min(matched_bbox[:, 1])
            x_max = max(matched_bbox[:, 0])
            y_max = max(matched_bbox[:, 1])
            scene_box = (x_min, y_min, x_max - x_min, y_max-y_min)
        else:
            # if false to match features, use simple scene_box
            scene_box = (x_new, y_new, w_new, h_new)
        self._add_new_feat_pts(scene_box)

    def _add_new_feat_pts(self, bbox: Tuple[int|float]):
        """
        Extracts and filters new feature points within the given bounding box.

        Args:
            bbox (tuple): Bounding box (x, y, w, h) used to define region for new feature detection.
        """
        new_points = self._extrac_feat_pts(bbox)
        if new_points is None:
            return
        existing_points = self.main_feat_pts.reshape(-1, 2) if self.main_feat_pts is not None else np.empty((0, 2))
        filtered_pts = []

        for pt in new_points.reshape(-1, 2):
            if len(existing_points) == 0:
                filtered_pts.append(pt)
                existing_points = np.array([pt])
            else:
                distances = np.linalg.norm(existing_points - pt, axis=1)
                if np.all(distances > 7):
                    filtered_pts.append(pt)
                    existing_points = np.vstack([existing_points, pt])
        if len(filtered_pts) > 0:
            self.prob_feat_pts = np.array(filtered_pts, dtype=np.float32).reshape(-1, 1, 2)

    def _matching(self, scene: np.ndarray) -> np.ndarray|None:
        """
        Matches ORB descriptors between the last good crop and current scene region.
        Estimates the object's transformation using homography.

        Args:
            scene (np.ndarray): Current grayscale scene crop to match against.

        Returns:

            np.ndarray or None: Transformed corner coordinates (4, 2) if matching is successful, else None.
        """
        kp1, des1 = self.descriptor.detectAndCompute(self.last_good_crop, None)
        kp2, des2 = self.descriptor.detectAndCompute(scene, None)
        if des1 is None or des2 is None:
            return
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # extract the matching keyspoints
        obj_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        scene_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Homography matrix
        if len(obj_pts) < 4 or len(scene_pts) < 4:
            return
        H, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 5.0)
        h, w = self.last_good_crop.shape
        obj_corners = np.float32([[0, 0],[0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        scene_corners = cv2.perspectiveTransform(obj_corners, H)
        return scene_corners.reshape(4, 2)

    def _upscale(self, bbox: Tuple[float|int], scale=1.):
        """
        Enlarges the bounding box by a scaling factor.

        Args:
            bbox (tuple): Original bounding box (x, y, w, h).
            scale (float): Scale factor for enlargement.

        Returns:
            tuple: Scaled bounding box (x_new, y_new, w_new, h_new).
        """
        x, y, w, h = bbox
        height, width = self.gray.shape

        new_w = w * scale
        new_h = h * scale
        new_x = x - (new_w - w) / 2
        new_y = y - (new_h - h) / 2

        x_new = int(max(0, min(new_x, width - 1)))
        y_new = int(max(0, min(new_y, height - 1)))
        x_max = int(min(new_x + new_w, width))
        y_max = int(min(new_y + new_h, height))
        w_new = max(1, x_max - x_new)
        h_new = max(1, y_max - y_new)

        return x_new, y_new, w_new, h_new

    def _get_bbox_from_pts(self):
        """
        Calculates the bounding box surrounding the current set of main feature points.

        Returns:
            tuple: Bounding box (x, y, w, h) based on min/max of feature coordinates.
        """
        x1, y1, x2, y2 = min(self.main_feat_pts[:, 0, 0]), min(self.main_feat_pts[:, 0, 1]), max(self.main_feat_pts[:, 0, 0]), max(self.main_feat_pts[..., 1])
        return (x1.item(), y1.item(), x2.item() - x1.item(), y2.item() - y1.item())

    def _filter_velocity(self, old_pair: Tuple[np.ndarray], new_pair: Tuple[np.ndarray]) -> Tuple[np.ndarray|None]:
        """
        Filters feature points based on motion direction consistency.

        Args:
            old_pair (tuple): Tuple of (main_old_pts, prob_old_pts).
            new_pair (tuple): Tuple of (main_new_pts, prob_new_pts).

        Returns:
            tuple: Indices of points to keep for main and probable features.
        """
        good_old, good_old_prob = old_pair
        good_new, good_new_prob = new_pair

        # first filter for self.main_feat_pts
        main_velocities = good_new - good_old
        norm_main_velocities = np.linalg.norm(main_velocities, axis=1, keepdims=True) + 1e-6
        main_directions = main_velocities/norm_main_velocities
        first_cost_matrix = np.dot(main_directions, main_directions.T)
        mean_similarity = np.mean(first_cost_matrix, axis=1)
        keep_mask = mean_similarity >= self.dist_similarity
        first_keep_ind = np.where(keep_mask)[0]

        # second filter for self.prob_feat_pts
        if good_old_prob is not None and good_new_prob is not None:
            prob_v = good_new_prob - good_old_prob
            norm_prob_v = np.linalg.norm(prob_v, axis=1, keepdims=True) + 1e-6
            prob_directions = prob_v/norm_prob_v
            second_cost_matrix = np.dot(prob_directions, main_directions.T)
            mean_similarity = np.mean(second_cost_matrix, axis=1)
            keep_mask = mean_similarity >= self.dist_similarity
            second_keep_ind = np.where(keep_mask)[0]
        else:
            second_keep_ind = None
        return (first_keep_ind, second_keep_ind)

    def _extrac_feat_pts(self, bbox: Tuple[int|float]):
        """
        Extracts Shi-Tomasi corner features from a specified bounding box region.

        Args:
            bbox (tuple): Bounding box (x, y, w, h) in which to detect features.

        Returns:
            np.ndarray or None: Feature points (N, 1, 2) or None if detection fails.
        """
        x1, y1, w, h = bbox
        crop = self.gray[int(y1): int(y1+h), int(x1):int(x1+w)]
        keypoints = cv2.goodFeaturesToTrack(crop, mask=None, **self.feature_params)
        if keypoints is not None and len(keypoints) > 0:
            keypoints[:, 0, 0] += x1
            keypoints[:, 0, 1] += y1
        return keypoints

    def get_mask(self):
        return self.mask.copy()

    def get_points(self):
        p0_int = self.main_feat_pts.astype(np.int32)
        return [tuple(pt[0]) for pt in p0_int]
