import random
import numpy as np

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Each box is represented as (x, y, width, height).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x_min, y_min, x_max, y_max)
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # Compute intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute intersection area
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Compute union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    # Compute IoU
    return inter_area / union_area if union_area > 0 else 0

def generate_bounding_boxes(original_box, image_size, min_IoU, num_boxes=20):
    """
    Generate new bounding boxes with IoU greater than min_IoU compared to the original box.
    """
    x_orig, y_orig, w_orig, h_orig = original_box
    img_width, img_height = image_size
    new_boxes = []

    while len(new_boxes) < num_boxes:
        # Randomly perturb the original bounding box
        new_x = max(0, x_orig + random.randint(-w_orig // 2, w_orig // 2))
        new_y = max(0, y_orig + random.randint(-h_orig // 2, h_orig // 2))
        new_w = max(5, min(img_width - new_x, w_orig + random.randint(-w_orig // 4, w_orig // 4)))
        new_h = max(5, min(img_height - new_y, h_orig + random.randint(-h_orig // 4, h_orig // 4)))

        new_box = (new_x, new_y, new_w, new_h)
        iou = compute_iou(original_box, new_box)

        if iou >= min_IoU:
            new_boxes.append(new_box)

    return new_boxes

# Example usage
original_box = (50, 50, 100, 100)  # (x, y, width, height)
image_size = (500, 500)  # (image_width, image_height)
min_IoU = 0.5

new_boxes = generate_bounding_boxes(original_box, image_size, min_IoU)
print(new_boxes)
