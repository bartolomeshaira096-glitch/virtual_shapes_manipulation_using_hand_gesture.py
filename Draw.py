import cv2
import numpy as np
import mediapipe as mp
import math

#    MACHINE LEARNING SHAPE CLASSIFIER

class ShapeClassifier:
    def __init__(self):
        self.knn = cv2.ml.KNearest_create()
        self.trained = False
        print("Training ML Shape Model...")
        self.train_model()
        print("Model Trained.")

    def get_features(self, contour):
        """ Extracts numerical features describing the shape """
        # 1. Hu Moments (7 features) - Invariant to Scale, Rotation, Translation
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        # Log scaling to make Hu moments comparable and usable
        hu_moments = [-1 * math.copysign(1.0, h) * math.log10(abs(h)) if h != 0 else 0 for h in hu_moments]
        
        # 2. Geometric Properties
        area = moments['m00']
        if area == 0: return None
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: return None

 # Circularity (0 to 1)
        circularity = 4 * math.pi * area / (perimeter * perimeter)

        # Solidity (Area / Convex Hull Area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Extent (Area / Bounding Rect Area)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        # Aspect Ratio (Width / Height)
        aspect_ratio = float(w)/h if h > 0 else 0
        # Normalize aspect ratio to always be <= 1 (min/max) for rotation invariance
        if aspect_ratio > 1: aspect_ratio = 1.0 / aspect_ratio

        # WEIGHTING: Scale geometric features to match range of Hu Moments (approx 0-10)
        # This ensures KNN treats geometric properties as important as Hu Moments
        geom_features = [circularity * 5, solidity * 5, extent * 5, aspect_ratio * 5]

        # Combine all features
        features = hu_moments + geom_features
        return np.array(features, dtype=np.float32)
    
    def generate_synthetic_data(self):
        """ Generates noisy shapes for training """
        samples = []
        labels = []
        
        # Helper to make features from points
        def process(pts, label):
            # Increase noise standard deviation (4.0) to simulate shaky hands
            noise = np.random.normal(0, 4.0, pts.shape).astype(np.int32)
            pts_noisy = pts + noise
            feat = self.get_features(pts_noisy)
            if feat is not None:
                samples.append(feat)
                labels.append(label)

    # 1. CIRCLE (Label 0)
        for _ in range(100):
            t = np.linspace(0, 2 * np.pi, 100)
            r = np.random.randint(40, 120)
            cx, cy = np.random.randint(100, 500), np.random.randint(100, 500)
            scale_x = np.random.uniform(0.9, 1.1)
            scale_y = np.random.uniform(0.9, 1.1)
            x = r * scale_x * np.cos(t) + cx
            y = r * scale_y * np.sin(t) + cy
            pts = np.stack((x, y), axis=1).astype(np.int32)
            process(pts, 0)

        # 2. TRIANGLE (Label 1)
        for _ in range(100):
            scale = np.random.randint(40, 120)
            cx, cy = np.random.randint(100, 500), np.random.randint(100, 500)
            pt1 = [0, -scale]
            pt2 = [-scale, scale]
            pt3 = [scale, scale]
            pts = np.array([pt1, pt2, pt3]) + [cx, cy]
            process(pts.astype(np.int32), 1)

        # 3. QUAD (RECT/SQUARE) (Label 2)
        for _ in range(100):
            w, h = np.random.randint(40, 120), np.random.randint(40, 120)
            cx, cy = np.random.randint(100, 500), np.random.randint(100, 500)
            pts = np.array([[0,0], [w,0], [w,h], [0,h]]) + [cx, cy]
            process(pts.astype(np.int32), 2)

        # 4. HEART (Label 3)
        for _ in range(100):
            t = np.linspace(0, 2 * np.pi, 100)
            scale = np.random.randint(5, 15)
            cx, cy = np.random.randint(100, 500), np.random.randint(100, 500)
            x = 16 * np.sin(t)**3
            y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t))
            pts = np.stack((x*scale, y*scale), axis=1).astype(np.int32) + [cx, cy]
            process(pts, 3)

        return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int32)
    def train_model(self):
        train_data, train_labels = np.array([]), []
        try:
           train_data, train_labels = self.shape_data if hasattr(self, 'shape_data') else self.generate_synthetic_data()
           self.knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
           self.trained = True
        except Exception as e:
           print(f"Failed to train KNN Model: {e}")
           self.trained = False


    def predict(self, contour):
        if not self.trained: return None, 9999
        feat = self.get_features(contour)
        if feat is None: return None, 9999
        
        feat_array = np.array([feat], dtype=np.float32)
        ret, results, neighbours, dist = self.knn.findNearest(feat_array, k=5)
        confidence_score = np.mean(dist)
        return int(results[0][0]), confidence_score

