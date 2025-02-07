import numpy as np
import cv2

from typing import Tuple

from kalman_filter import KalmanFilter
from appearence_model import PatchBox, AppearanceModel, CandidateSet

class Tracker:
    def __init__(self, bbox: Tuple[int], control_parameters: dict):
        self.control_parameters = control_parameters
        self.obj = PatchBox(bbox)
        self.appearance_model = AppearanceModel(
            PP_threshold=control_parameters['PP_threshold'],
            NP_threshold=control_parameters['NP_threshold'],
        )
        self.motion_model = KalmanFilter(init_bbox=bbox)

    def track(self, TD_threshold, frame: np.ndarray):

        self.appearance_model.generate_patches(frame=frame, obj=self.obj)
        patches = self.appearance_model.get_patches()

        candidate_set = CandidateSet(
            obj=self.obj,
            frame=frame,
            CA_threshold=self.control_parameters['CA_threshold']
            )
        candidate_set.calculate_CS_weak(
            patches=patches,
            WC_threshold=self.control_parameters['WC_threshold']
        )

        pred_key, pred_weak_bbox = candidate_set.predict_bbox()
        pred_motion_bbox = self.motion_model.predict()
        if pred_weak_bbox is not None:
            if pred_motion_bbox is not None:
                TD = tolerabel_distance(pred_motion_bbox, pred_weak_bbox)
                if TD < TD_threshold:
                    self.motion_model.update(pred_motion_bbox)
                    optimal_bbox = self.motion_model.get_bbox()
                    self.obj.update(optimal_bbox)
                else:
                    candidate_set.calculate_CS_strong(self.control_parameters['SC_threshold'])
                    pred_strong_bbox = candidate_set.predict_bbox_strong(except_key=pred_key)
                    if pred_strong_bbox is not None:
                        self.motion_model.update(pred_strong_bbox)
                        optimal_bbox = self.motion_model.get_bbox()
                        self.obj.update(optimal_bbox)
            else:
                self.motion_model = KalmanFilter(init_bbox=pred_weak_bbox) # reset motion tracker
                self.obj.update(pred_weak_bbox)
        elif pred_motion_bbox is not None:
            self.obj.update(pred_motion_bbox)
            self.appearance_model.del_pos_patches()
        else:
            candidate_set.calculate_CS_strong(self.control_parameters['SC_threshold'])
            pred_strong_bbox = candidate_set.predict_bbox_strong(except_key=pred_key)
            if pred_strong_bbox is not None:
                self.motion_model = KalmanFilter(init_bbox=pred_strong_bbox) # reset motion tracker
                self.obj.update(pred_strong_bbox)
            else:
                print('Fail tracking')

def tolerabel_distance(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    w_avg = float(w1 - w2)
    h_avg = float(h1 - h2)
    return dist/(w_avg + h_avg)

Parameters = {
    'WC_threshold': 0.47,   # Threshold score for a weak confidence candidate decision
    'TD_threshold': 0.69,   # Tolerable distance between the motion- and model-based trackers
    'CA_threshold': 0.63,   # Threshold for a candidate area decision
    'NP_threshold': 0.70,   # Threshold for a negative patch determination
    'PP_threshold': 0.38,   # Threshold for a positive patch determination
    'SC_threshold': 0.61    # Threshold score for a strong confidence candidate decision
}