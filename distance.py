import numpy as np
import cv2
import os
class Distance:
    def __init__(self, type='cosine', size=(8, 8)):
        self.type = type
        self.size = size
    def __call__(self, image, list_image: list):
        if self.type == 'cosine':
            return self.compute_cosine(image, list_image)
        elif self.type == 'euclidean':
            return self.compute_euclidean(image, list_image)
        elif self.type == 'sift':
            return self.compute_sift(image, list_image)
        elif self.type == 'orb':
            return self.compute_orb(image, list_image)
        # elif self.type == 'surf':
        #     return self.compute_surf(image, list_image)

    def compute_cosine(self, image, list_image: list):
        vect1 = self.preprocess(image)
        norm1 = np.linalg.norm(vect1, keepdims = True)
        if norm1 == 0.:
            return np.zeros((len(list_image)))
        vects = [self.preprocess(i) for i in list_image]
        vects = np.stack(vects)
        norms = np.linalg.norm(vects, axis=1, keepdims=True)
        vects_normalized = vects / np.where(norms == 0, 1, norms)
        vect1_normalized = vect1 / norm1  # shape (64,)
        similarities = np.dot(vects_normalized, vect1_normalized)
        return similarities

    def compute_euclidean(self, image, list_image: list):
        similarities = [np.linalg.norm(image - i) for i in list_image]
        return np.stack(similarities)
    def preprocess(self, image):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = image[:,:, 2]
        gray = cv2.resize(gray, self.size)
        gray = gray.flatten().astype(np.float32)
        return gray/255.0

    def compute_sift(self, image, list_images, n_features=500, distance_thershold=100):
        sift = cv2.SIFT_create()
        def _per_image(img1, img2):
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            if des1 is None or des2 is None:
                return 0.
            
            flann = cv2.FlannBasedMatcher()
            matches = flann.knnMatch(des1, des2, k = 2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            similarity = len(good_matches)/ min(len(kp1), len(kp2))
            return similarity
        return np.array([_per_image(image, img2) for img2 in list_images])
    
    def compute_orb(self, image, list_images, n_features=500, distance_thershold=50):
        orb = cv2.ORB_create(nfeatures=n_features)
        def _per_image(img1, img2):
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            if des1 is None or des2 is None:
                return 0.

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            good_matches = [m for m in matches if m.distance < distance_thershold]
            similarity = len(good_matches)/ min(len(kp1), len(kp2))
            return similarity
        return np.array([_per_image(image, img2) for img2 in list_images])
    
    def compute_surf(self, image, list_images, hessian_threshold=100):
        surf = cv2.xfeatures2d.SURF_create()
        def _per_image(img1, img2):
            kp1, des1 = surf.detectAndCompute(img1, None)
            kp2, des2 = surf.detectAndCompute(img2, None)
            if des1 is None or des2 is None:
                return 0.
            flann = cv2.FlannBasedMatcher()
            matches = flann.knnMatch(des1, des2, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            similarity = len(good_matches) / min(len(kp1), len(kp2))
            return similarity
        return np.array([_per_image(image, img2) for img2 in list_images])




folder = r'C:\Users\ABC\Desktop\projects\AdaptiveVisualTracking\video'
imgs = os.listdir(folder)
imgs.sort()

image_paths = [os.path.join(folder, img) for img in imgs[: 10]]

distance = Distance('sift')
img = cv2.imread(os.path.join(folder, imgs[15]))
list_of_images = [cv2.imread(img) for img in image_paths]
print(distance(img, list_of_images))

