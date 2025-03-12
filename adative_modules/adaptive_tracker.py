import numpy as np

from typing import Tuple

from kalman_filter import KalmanFilter
from appearence_model import PatchBox, AppearanceModel, CandidateSet

class MyTracker:
    def __init__(self, control_parameters: dict):
        self.control_parameters = control_parameters
        self.obj = None
        self.appearance_model = AppearanceModel(
            PP_threshold=control_parameters['PP_threshold'],
            NP_threshold=control_parameters['NP_threshold'],
        )
        self.motion_model = None
    def init(self, frame, init_bbox_roi: Tuple[int]):
        """
        init_bbox_roi is (x1, y1, w, h)
        """
        xc = init_bbox_roi[0] + init_bbox_roi[2]//2
        yc = init_bbox_roi[1] + init_bbox_roi[3]//2
        bbox = (xc, yc, init_bbox_roi[2], init_bbox_roi[3])
        self.obj = PatchBox(bbox)
        self.motion_model = KalmanFilter(bbox)
        self.appearance_model.generate_patches(frame=frame, obj=self.obj)   # Need to check
    def update(self, frame: np.ndarray):

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
                if TD < self.control_parameters['TD_threshold']:
                    self.motion_model.update(pred_motion_bbox)
                    optimal_bbox = self.motion_model.get_bbox()
                    self.obj.update(optimal_bbox)
                else:
                    candidate_set.calculate_CS_strong(
                        patches=patches,
                        SC_threshold=self.control_parameters['SC_threshold'])
                    pred_strong_bbox = candidate_set.predict_bbox_strong(except_key=pred_key)
                    if pred_strong_bbox is not None:
                        self.motion_model.update(pred_strong_bbox)
                        optimal_bbox = self.motion_model.get_bbox()
                        self.obj.update(optimal_bbox)
            else:
                self.motion_model = KalmanFilter(bbox=pred_weak_bbox) # reset motion tracker
                self.obj.update(pred_weak_bbox)
        elif pred_motion_bbox is not None:
            self.obj.update(pred_motion_bbox)
            self.appearance_model.del_pos_patches()
        else:
            candidate_set.calculate_CS_strong(
                patches=patches,
                SC_threshold=self.control_parameters['SC_threshold'])
            pred_strong_bbox = candidate_set.predict_bbox_strong(except_key=pred_key)
            if pred_strong_bbox is not None:
                self.motion_model = KalmanFilter(init_bbox=pred_strong_bbox) # reset motion tracker
                self.obj.update(pred_strong_bbox)
            else:
                print('Fail tracking')
                return False, None
        return True, self.obj.to_bbox_roi()

def tolerabel_distance(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    w_avg = float(w1 - w2)
    h_avg = float(h1 - h2)
    return dist/(w_avg + h_avg)
