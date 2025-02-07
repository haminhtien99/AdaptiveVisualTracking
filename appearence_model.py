import numpy as np
import cv2
import random
from typing import Tuple
from abc import ABC, abstractmethod

W, H = 100, 100
class BoundingBox:
    """
    Class represent bounding box
    """
    def __init__(self, bbox: Tuple[int]):
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
    def to_xywh(self):
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]
    def to_center_xywh(self):
        xc, yc = (self.x1 + self.x2)//2, (self.y1 + self.y2)//2
        return [xc, yc, self.x2 - self.x1, self.y2 - self.y1]
    def iou_rate(self, bbox):
        """
        Calculate Intersection over Union (IoU) rate of two bounding boxes
        """
        xx1 = max(self.x1, bbox[0])
        yy1 = max(self.y1, bbox[1])
        xx2 = min(self.x2, bbox[2])
        yy2 = min(self.y2, bbox[3])
        overlap = (xx2 - xx1) * (yy2 - yy1)
        area1 = (self.x2 - self.x1) * (self.y2 - self.y1)
        area2 = (bbox[2] - bbox[0]) * (bbox[3]-bbox[1])
        iou = overlap / (area1 + area2 - overlap)
        return iou

class PatchBox:
    """
    Class represent object patch from bbox
    """
    def __init__(self, bbox: Tuple[int]):
        """
        bbox - [xc, yc, w, h]
        """
        self.xc, self.yc, self.w, self.h = bbox
        self.theta = 0.
        self.state = np.array([self.xc, self.yc, 1.0, 1.0, 0.])
    def distance_state(self, obj2_state, weights: Tuple[int] = [0.5, 0.4, 0.1]):
        delta = self.state - obj2_state
        weights = np.array(weights)
        norms = np.array([np.linalg.norm(delta[:2]),
                          np.linalg.norm(delta[2:4]),
                          abs(delta[-1])])
        return np.dot(weights, norms)
    def update(self, new_bbox):
        theta = self.theta
        theta += np.arctan2(new_bbox[1] - self.yc, new_bbox[0] - self.xc)
        theta /= 2
        if theta < 0: theta = np.pi - theta
        self.state = [new_bbox[0], new_bbox[1], new_bbox[2]/ self.w, new_bbox[3]/self.h, theta]

class AppearanceModel:
    def __init__(
        self,
        PP_threshold: float,
        NP_threshold: float,
        max_len=10
        ):
        self.NP_threshold = NP_threshold
        self.PP_threshold = PP_threshold
        self.max_len = max_len
        self.pos_patches = None
        self.neg_patches = None
    def generate_patches(self, frame: np.ndarray, obj: PatchBox):
        """
        Generates patches, filtering out patches
        """
        # crop the patches
        cropped_pos, cropped_neg = self.generate_cropped_patches(frame, obj)
        # warp Transformation
        warp_transform_pos = warp_transform(cropped_pos)
        warp_transform_neg = warp_transform(cropped_neg)
        # Filter patches
        self.filter(
            new_negative_patches=warp_transform_pos,
            new_negative_patches=warp_transform_neg
            )
    def generate_cropped_patches(self, frame: np.ndarray, obj: PatchBox):
        # generate max_len crop images from the optimal state
        optimal_state = obj.state
        w, h = obj.w, obj.h
        cropped_pos = []
        cropped_neg = []
        while len(cropped_pos) < self.max_len:
            dx, dy = random.randint(-w/2, w/2 + 1, w/10), random.randint(-h/2, h/2 + 1, h/10)
            x, y = optimal_state[:2] + np.array([dx, dy])
            ax, ay = random.uniform(0.5, 1.2), random.uniform(0.5, 1.2)
            ax = ax * optimal_state[2]
            ay = ay * optimal_state[3]
            angel = min(max(optimal_state[4] + random.uniform(-0.3, 0.3), 0), 2 * np.pi)

            possiable_state = np.array([x, y, ax, ay, angel])
            dist = obj.distance_state(possiable_state)
            if dist < self.PP_threshold:
                crop = crop_image_from_state(possiable_state, obj, frame)
                cropped_pos.append(crop)

        while len(cropped_pos) < self.max_len:
            dx, dy = random.randint(-w, w + 1, w/5), random.randint(-h, h + 1, h/5)
            x, y = optimal_state[:2] + np.array([dx, dy])
            ax, ay = random.uniform(0.3, 1.5), random.uniform(0.3, 1.5)
            ax = ax * optimal_state[2]
            ay = ay * optimal_state[3]
            angel = min(max(optimal_state[4] + random.uniform(-0.5, 0.5), 0), 2 * np.pi)

            possiable_state = np.array([x, y, ax, ay, angel])
            dist = obj.distance_state(possiable_state)
            if dist < self.NP_threshold and dist > self.PP_threshold:
                crop = crop_image_from_state(possiable_state, obj, frame)
                cropped_neg.append(crop)
        return np.array(cropped_pos), np.array(cropped_neg)
    def filter(self, new_positive_patches, new_negative_patches):
        criteri = ''
        # Update patches with some criteria
        if self.pos_patches is None:
            self.pos_patchs = new_positive_patches
        elif self.neg_patches is None:
            self.neg_patches = new_negative_patches
        else:
            pass
    def get_patches(self):
        return [self.pos_patches, self.neg_patches]
    def del_pos_patches(self):
        self.pos_patches = None

def crop_image_from_state(state, object: PatchBox, image: np.ndarray):
    w, h = object.to_wh()
    x, y, ax, ay, _ = state
    x1 = int(max(0, x - w * ax /2))
    y1 = int(max(0, y - h * ay /2))
    x2 = int(min(x + w * ax /2, image.shape[1]))
    y2 = int(min(y + h * ay /2, image.shape[0]))
    image_crop = image[y1: y2, x1: x2]
    return image_crop

def warp_transform(images, g=(0.2, 0.1, 0.5, -0.3)):
    g1, g2, g3, g4 = g
    M_affine = np.array(
        [[1 + g1, -g2, g3],
        [g2, 1 + g1, g4]],
        dtype=np.float32
        )
    warp_images = np.zeros_like(images, dtype=np.int8)
    for i, img in enumerate(images):
        warped = cv2.warpAffine(
            img,M_affine,
            (img.shape[1],img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=(0,0,0)
            )
        warp_images[i] = warped
    return warp_images


class CandidateSet:
    def __init__(self, obj, frame: np.ndarray, CA_threshold: float):
        self.init_canndidate_set = self.init_set(obj, CA_threshold, frame)  # includes bbox information and crop_image
        self.CS_weak = None # the confidence score
        self.C_weak = {}  # the weak condidate set dictionary of pairs (id, CS_weak)
        self.CS_weak = None
        self.C_strong = {}
    def init_set(self, obj: PatchBox, CA_threshold, frame: np.ndarray):
        # TODO: implement initial candidate set generation based on object and returns candidate set
        pass
    def calculate_CS_weak(self, patches: np.ndarray, WC_threshold):
        "Calculates the weak confience score"
        # resize all patches to [w, h, ...]
        # Convert to [0, 1] float32
        pos_patches, neg_patches = patches
        converted_pos_patches = self.convert_patch(pos_patches)
        converted_neg_patches = self.convert_patch(neg_patches)
        converted_candidate_patches = self.convert_patch(self.init_canndidate_set['patches'])
        # Calculate confidence score
        self.CS_weak = np.zeros(len(self.init_canndidate_set), dtype=np.float32)
        for i, candidate in enumerate(converted_candidate_patches):
            INV_pos = kNN_regression(candidate, converted_pos_patches)
            INV_neg = kNN_regression(candidate, converted_neg_patches)
            self.CS_weak[i] = INV_pos / (INV_pos + INV_neg)

        # Find weak candidate set
        for i, CS in enumerate(self.CS_weak):
            if CS > WC_threshold:
                bbox = self.init_canndidate_set['bbox'][i]
                self.C_weak[i] = bbox
    
    def calculate_CS_strong(self, patches: np.ndarray, SC_threshold: float):
        pos_patches, neg_patches = patches
        converted_pos_patches = self.convert_patch(pos_patches)
        converted_neg_patches = self.convert_patch(neg_patches)
        converted_candidate_patches = self.convert_patch(self.init_canndidate_set['patches'])
        # Calculate confidence score
        self.CS_strong = np.zeros(len(converted_candidate_patches))
        for i, candidate in enumerate(converted_candidate_patches):
            INV_neg = kNN_regression(candidate, converted_neg_patches)
            INV_pos = kNN_regression_55_percent(candidate, converted_pos_patches)
            self.CS_strong = INV_pos / (INV_pos + INV_neg)
        # Find strong candidate set
        for i, CS in enumerate(self.CS_strong):
            if CS > SC_threshold:
                bbox = self.init_canndidate_set['bbox'][i]
                self.C_strong[i] = bbox
    def convert_patch(self, patches):
        # Convert patches to [0, 1] float32
        converted_patches = np.zeros_like(patches)
        for i, patch in enumerate(patches):
            patch = cv2.resize(patch, (self.w, self.h))
            patch = patch / 255.0
            converted_patches[i] = patch
        return converted_patches
    def predict_bbox(self):
        if len(self.C_weak) > 0:

            key_max = np.argmax(self.CS_weak)
            bbox_weak = self.C_weak[key_max]

            return key_max, bbox_weak
        else:
            return None, None
    def predict_bbox_strong(self, except_key: int|None=None):
        if len(self.C_strong) > 0:
            # Remove the bounding box of except_key
            if except_key is not None and except_key in self.C_strong.keys():
                self.C_strong = self.C_strong.pop(except_key)
            key_max = np.argmax(self.CS_strong)
            bbox_strong = self.C_strong[key_max]

            return bbox_strong
def kNN_regression(candidate: np.ndarray, patches: np.ndarray, k = 5):
    # Compute the Euclidean distance between candidate and each patch
    squared_diff = (patches - candidate) ** 2
    sum_squared = np.sum(squared_diff, axis=(1,2,3))
    distances = np.sqrt(sum_squared)
    # Get the k nearest neighbor
    distances.sort()
    k_nearest_distaces = distances[:k]
    average_distance = k_nearest_distaces.sum()/ k
    return 1 / average_distance

def kNN_regression_55_percent(candidate: np.ndarray, pos_patches: np.ndarray):
    # the inverse of the average distance of 55% of high-ranked positive neighbors
    k = int(0.55) * len(pos_patches)
    return kNN_regression(candidate=candidate, patches=pos_patches, k=k)

def init_candidate_set(opt_obj: PatchBox, CA_threshold):
    xc, yc, w, h = opt_obj.bbox.to_center_xywh()
    # Given a bounding box and an IoU threshold,
    # find a bounding box within the original bounding box that satisfies that IoU
    generated_boxes = []
    
    # Maximum pixel difference of the subbox from the original box
    max_pixel_diff = [0]
    thresholds = np.arange(CA_threshold, 1., 0.1)
    for threshold in thresholds:
        diff = int((h+w)/2 - np.sqrt((h+w)**2 /4 - (1- threshold)* h*w/(1+ threshold))) + 1
        max_pixel_diff.append(diff)
    # Generate bounding-boxes with different between its own center and center of original bounding box 
    # less than max_pixel_diff
    for i, diff in enumerate(max_pixel_diff[1:]):
        new_centers = []
        delta = random.randint(max_pixel_diff[i - 1], diff)
        for i in [-delta, delta]:
            for j in [-delta, delta]:
                xc2, yc2 = xc + i, yc + j
                new_centers.append([xc2, yc2])
        for i in [-2*delta, 2*delta]:
            xc2, yc2 = xc + i, yc
            new_centers.append([xc2, yc2])
        for j in [-2*delta, 2*delta]:
            xc2, yc2 = xc, yc + j
            new_centers.append([xc2, yc2])
        for center in new_centers:
            x21, y21 = center[0] - w/2, center[1] - h/2
            x22, y22 = center[0] + w/2, center[1] + h/2
            bbox = [x21, y21, x22, y22]
            iou = opt_obj.iou_rate(bbox)
            if iou > threshold:
                generated_boxes.append(bbox)
        bbox = [xc - w/2 - delta/2, yc - h/2 - delta/2, xc + w/2 - delta/2, yc + h/2 - delta]
        iou = opt_obj.iou_rate(bbox)
        generated_boxes.append(bbox)
    candidate_images = []
    for bbox in generated_boxes:
        x21, y21, x22, y22 = bbox
        image_crop = opt_obj.image[int(y21):int(y22), int(x21):int(x22)]
        candidate_images.append(image_crop)
    return np.array(candidate_images)
