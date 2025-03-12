import numpy as np
import cv2
import random
from typing import Tuple
from collections import deque
from abc import ABC, abstractmethod
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
        x1, y1, ax1, ay1, theta1 = self.state
        x2, y2, ax2, ay2, theta2 = obj2_state
        xy = np.array([x1, y1])
        delta_xy = np.array([x1 - x2, y1 - y2])
        delta_axy = np.array([ax1 - ax2, ay1 - ay2])
        norm1 = np.linalg.norm(delta_xy) / np.linalg.norm(xy)
        norm2 = np.linalg.norm(delta_axy) / 1.414
        norm3 = abs(theta1 - theta2) / 2 * np.pi
        return np.dot(weights, [norm1, norm2, norm3])
    def update(self, new_bbox):
        theta = self.theta
        theta += np.arctan2(new_bbox[1] - self.yc, new_bbox[0] - self.xc)
        theta /= 2
        if theta < 0: theta = 2* np.pi + theta
        if theta > 2 * np.pi: theta = theta - 2 * np.pi
        self.xc = int(new_bbox[0])
        self.yc = int(new_bbox[1])
        self.state = np.array([self.xc, self.yc, new_bbox[2]/self.w, new_bbox[3]/self.h, theta])
    def to_bbox_roi(self):
        x, y, ax, ay, _ = self.state
        x1 = int(x - ax*self.w/2)
        y1 = int(y - ay*self.h/2)
        w = ax * self.w
        h = ay * self.h
        return (x1, y1, w, h)


class AppearanceModel:
    def __init__(
        self,
        PP_threshold: float=0.38,
        NP_threshold: float=0.70,
        max_len=5,
        warp_g: Tuple[float] = (0.2, 0.1, 0.5, -0.3)
        ):
        self.NP_threshold = NP_threshold
        self.PP_threshold = PP_threshold
        self.max_len = max_len
        self.pos_patches = deque(maxlen=4)
        self.neg_patches = deque(maxlen=4)
        g1, g2, g3, g4 = warp_g
        self.M_affine = np.array(
            [[1 + g1, -g2, g3],
             [g2, 1 + g1, g4]],
            dtype=np.float32
        )
    def generate_patches(self, frame: np.ndarray, obj: PatchBox):
        """
        Generates patches, filtering out patches
        """
        # crop the patches
        cropped_pos, cropped_neg = self._generate_cropped_patches(frame=frame, obj=obj)
        # warp Transformation
        warp_pos = [cv2.warpAffine(img, self.M_affine, (img.shape[1], img.shape[0]), 
                    flags=cv2.INTER_LINEAR, borderValue=(0,0,0)) for img in cropped_pos]
        warp_neg = [cv2.warpAffine(img, self.M_affine, (img.shape[1], img.shape[0]), 
                    flags=cv2.INTER_LINEAR, borderValue=(0,0,0)) for img in cropped_neg]
        # Filter patches
        self._filter(warp_pos, warp_neg)
    def _generate_cropped_patches(self, frame: np.ndarray, obj: PatchBox):
     # generate max_len crop images from the optimal state
        optimal_state = obj.state
        w, h = obj.w, obj.h
        cropped_pos = []
        cropped_neg = []
        while len(cropped_pos) < self.max_len:
            dx = random.randint(w//5, w//3) * random.choice([-1, 1])
            dy = random.randint(h//5, h//3) * random.choice([-1, 1])
            x, y = optimal_state[:2] + np.array([dx, dy])
            ax = 1. - random.uniform(0.1, 0.5) * random.choice([-1, 1])
            ay = 1. - random.uniform(0.1, 0.5) * random.choice([-1, 1])
            ax = ax * optimal_state[2]
            ay = ay * optimal_state[3]
            theta = (optimal_state[-1] + np.arctan2(dy, dx))/ 2
            if theta < 0: theta = np.pi - theta
            if theta > 2 * np.pi: theta = theta - 2 * np.pi
            possiable_state = np.array([x, y, ax, ay, theta])
            dist = obj.distance_state(possiable_state)
            if dist < self.PP_threshold:
                crop = crop_image_from_state(possiable_state, obj, frame)
                cropped_pos.append(crop)

        while len(cropped_neg) < self.max_len:
            dx = random.randint(w//5, w//2) * random.choice([-1, 1])
            dy = random.randint(h//5, h//2) * random.choice([-1, 1])
            x, y = optimal_state[:2] + np.array([dx, dy])
            ax = 1. - random.uniform(0.5, 0.7) * random.choice([-1, 1])
            ay = 1. - random.uniform(0.5, 0.7) * random.choice([-1, 1])
            ax = ax * optimal_state[2]
            ay = ay * optimal_state[3]
            theta = (optimal_state[-1] + np.arctan2(dy, dx))/ 2
            if theta < 0: theta = np.pi - theta
            if theta > 2 * np.pi: theta = theta - 2 * np.pi
            possiable_state = np.array([x, y, ax, ay, theta])
            dist = obj.distance_state(possiable_state)
            if dist < self.NP_threshold and dist > self.PP_threshold:
                crop = crop_image_from_state(possiable_state, obj, frame)
                cropped_neg.append(crop)
        return cropped_pos, cropped_neg
    
    def _filter(self, new_pos, new_neg):
        self.pos_patches.append(new_pos)
        self.neg_patches.append(new_neg)

    def get_patches(self):
        flatten_pos_patches = [patch for patches in self.pos_patches for patch in patches]
        flatten_neg_patches = [patch for patches in self.neg_patches for patch in patches]
        return [flatten_pos_patches, flatten_neg_patches]

    def del_pos_patches(self):
        self.pos_patches.clear()

def crop_image_from_state(state, object: PatchBox, image: np.ndarray):
    w, h = object.w, object.h
    x, y, ax, ay, _ = state
    x1 = max(0, int(x - w * ax /2))
    y1 = max(0, int(y - h * ay /2))
    x2 = min(int(x + w * ax /2), image.shape[1])
    y2 = min(int(y + h * ay /2), image.shape[0])
    image_crop = image[y1: y2, x1: x2]
    return image_crop

def warp_transform(images, g=(0.2, 0.1, 0.5, -0.3)):
    g1, g2, g3, g4 = g
    M_affine = np.array(
        [[1 + g1, -g2, g3], 
        [g2, 1 + g1, g4]],
        dtype=np.float32
        )
    warp_images = []
    for i, img in enumerate(images):
        warped = cv2.warpAffine(
            img,M_affine,
            (img.shape[1],img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=(0,0,0)
            )
        warp_images.append(warped)
    return warp_images
class CandidateSet:
    def __init__(self, obj: PatchBox, frame: np.ndarray, CA_threshold: float):
        
        self.w = obj.w
        self.h = obj.h
        _, _, d = frame.shape
        self.num_candidates = 10
        self.init_boxes = np.zeros((self.num_candidates, 4), dtype=np.int32)
        self.init_img = np.zeros((self.num_candidates, self.h, self.w, d), dtype=np.float32)
        self.CS_weak = np.zeros(self.num_candidates, dtype=np.float32) # the confidence score
        self.CS_strong
        bbox = [obj.xc, obj.yc, obj.w, obj.h]
        bbox_xyxy = self._to_xyxy(bbox)
        self.init_set(bbox_xyxy, CA_threshold, frame)
        
    def _to_xyxy(self, original_box):
        xc, yc, w, h = original_box
        x1, y1 = xc - w/2, yc - h/2
        x2, y2 = xc + w/2, yc + h/2
        return [x1, y1, x2, y2]
    def init_set(self, xyxy, CA_threshold, frame):
        def generate_perturbed_box(original_box, alpha=0.1, beta=0.1):
            cx = (original_box[0] + original_box[2]) / 2
            cy = (original_box[1] + original_box[3]) / 2
            w = original_box[2] - original_box[0]
            h = original_box[3] - original_box[1]
            
            # Nhiễu tâm và kích thước
            delta_x = np.random.uniform(-alpha * w, alpha * w)
            delta_y = np.random.uniform(-alpha * h, alpha * h)
            new_w = w * np.random.uniform(1 - beta, 1 + beta)
            new_h = h * np.random.uniform(1 - beta, 1 + beta)
            
            # Tạo bounding box mới
            new_cx = cx + delta_x
            new_cy = cy + delta_y
            x1 = new_cx - new_w / 2
            y1 = new_cy - new_h / 2
            x2 = new_cx + new_w / 2
            y2 = new_cy + new_h / 2
            
            return [x1, y1, x2, y2]

        def generate_boxes(original_box, iou_threshold=0.7, num_boxes=10, max_attempts=100):
            generated_boxes = []
            attempts = 0
            while len(generated_boxes) < num_boxes and attempts < max_attempts:
                perturbed_box = generate_perturbed_box(original_box)
                iou = compute_iou(original_box, perturbed_box)
                if iou >= iou_threshold:
                    generated_boxes.append(perturbed_box)
                attempts += 1
            return generated_boxes

        def compute_iou(box1, box2):
            # Tính diện tích giao nhau và hợp nhất
            x1_inter = max(box1[0], box2[0])
            y1_inter = max(box1[1], box2[1])
            x2_inter = min(box1[2], box2[2])
            y2_inter = min(box1[3], box2[3])
            
            area_inter = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
            
            area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            area_union = area_box1 + area_box2 - area_inter
            
            return area_inter / area_union if area_union > 0 else 0
        bboxes = generate_boxes(xyxy, CA_threshold, self.num_candidates, 100)
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop_img = frame[y1: y2, x1: x2,:]
            resize_img = cv2.resize(crop_img, (self.w, self.h))
            self.init_img[i, :, :, :](resize_img)
            self.init_boxes[i, :] = np.array(box)

    def calculate_CS_weak(self, patches: list[list[np.ndarray]], WC_threshold):
        "Calculates the weak confience score"
        # resize all patches to [w, h, ...]
        # Convert to [0, 1] float32
        pos_patches, neg_patches = patches[0], patches[1]
        converted_pos_patches = self.convert_patch(pos_patches)
        converted_neg_patches = self.convert_patch(neg_patches)
        # Calculate confidence score
        for i, candidate in enumerate(self.init_img):
            INV_pos = kNN_regression(candidate, converted_pos_patches)
            INV_neg = kNN_regression(candidate, converted_neg_patches)
            self.CS_weak[i] = INV_pos / (INV_pos + INV_neg)

        # Find weak candidate set
        indices_C_weak = self.CS_weak > WC_threshold

    def calculate_CS_strong(self, patches: np.ndarray, SC_threshold: float):
        pos_patches, neg_patches = patches
        converted_pos_patches = self.convert_patch(pos_patches)
        converted_neg_patches = self.convert_patch(neg_patches)
        # Calculate confidence score
        self.CS_strong = np.zeros(self.num_candidates)
        for i, candidate in enumerate(self.init_img):
            INV_neg = kNN_regression(candidate, converted_neg_patches)
            INV_pos = kNN_regression_55_percent(candidate, converted_pos_patches)
            self.CS_strong[i] = INV_pos / (INV_pos + INV_neg)
        indices_C_strong = self.CS_strong > SC_threshold
    def convert_patch(self, patches, target_shape=(128,128)):
        # Convert patches to [0, 1] float32
        converted_patches = []
        for i, patch in enumerate(patches):
            patch = cv2.resize(patch, (self.w, self.h))
            patch = patch / 255.0
            converted_patches.append(patch)
        return np.array(converted_patches)
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






original_box = [50, 50, 150, 150]  # (x1, y1, x2, y2)
generated_boxes = generate_boxes(original_box, iou_threshold=0.7, num_boxes=10)
print(generated_boxes)
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
    k = int(0.55 * len(pos_patches))
    return kNN_regression(candidate=candidate, patches=pos_patches, k=k)

def compute_iou(bbox1: list[int], bbox2: list[int]):
    """
    Compute the IoU between two bounding boxes
    bbox [xc, yc, w, h]
    """
    xc1, yc1, w1, h1 = bbox1
    xc2, yc2, w2, h2 = bbox2
    x11, y11, x12, y12 = xc1 - w1/2, yc1 - h1/2, xc1 + w1/2, yc1 + h1/2
    x21, y21, x22, y22 = xc2 - w2/2, yc2 - h2/2, xc2 + w2/2, yc2 + h2/2
    inter_area = max(0, min(x12, x22) - max(x11, x21)) * max(0, min(y12, y22) - max(y11, y21))
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area/union_area

# if __name__ == "__main__":
#     import cv2
#     import os
#     frame = cv2.imread(os.path.join('src_video', '0001.jpg'))
#     model = AppearanceModel(PP_threshold=0.38, NP_threshold=0.70)
#     init_bbox = [808, 308, 80, 100]
#     patch_bbox1 = PatchBox(init_bbox)
#     model.generate_patches(frame=frame, obj=patch_bbox1)
    
#     frame2 = cv2.imread(os.path.join('src_video', '0002.jpg'))
#     model.generate_patches(frame=frame2, obj=patch_bbox1)
#     pos_patches, neg_patches = model.get_patches()
#     candidates = CandidateSet(patch_bbox1, frame2, 0.63)
#     candidates.calculate_CS_weak([pos_patches, neg_patches], 0.52)
#     _, bbox = candidates.predict_bbox()
#     if bbox is not None:
#         x1 = bbox[0] - bbox[2]//2
#         y1 = bbox[1] - bbox[3]//2
#         x2 = bbox[0] + bbox[2]//2
#         y2 = bbox[1] + bbox[3]//2
#         cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.imshow('frame2', frame2)
#         cv2.waitKey(0)
        
#     frame2 = cv2.imread(os.path.join('src_video', '0003.jpg'))
#     model.generate_patches(frame=frame2, obj=patch_bbox1)
#     pos_patches, neg_patches = model.get_patches()
#     candidates = CandidateSet(patch_bbox1, frame2, 0.63)
#     candidates.calculate_CS_weak([pos_patches, neg_patches], 0.52)
#     _, bbox = candidates.predict_bbox()
#     if bbox is not None:
#         x1 = bbox[0] - bbox[2]//2
#         y1 = bbox[1] - bbox[3]//2
#         x2 = bbox[0] + bbox[2]//2
#         y2 = bbox[1] + bbox[3]//2
#         cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.imshow('frame2', frame2)
#         cv2.waitKey(0)
    
    