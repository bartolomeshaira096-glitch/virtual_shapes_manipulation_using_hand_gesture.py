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

clf = ShapeClassifier()

#    STANDARD CANVAS LOGIC

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] 

    def findHands(self, img, draw=True):
        if img is None: return None
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        lmList = []
        if img is None: return []
        if self.results and self.results.multi_hand_landmarks and len(self.results.multi_hand_landmarks) > handNo:
            myHand = self.results.multi_hand_landmarks[handNo]
            if not getattr(myHand, 'landmark', None): return []
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList

    def fingersUp(self, lmList):
        fingers = []
        if len(lmList) < max(self.tipIds) + 1: return []
        if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]: fingers.append(1)
        else: fingers.append(0)
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]: fingers.append(1)
            else: fingers.append(0)
        return fingers
    
    class ShapeObject:
        """ Represents a floating interactive object """
    def __init__(self, shape_type, params, color):
        self.type = shape_type 
        self.params = params   
        self.color = color
        self.thickness = 3
        self.rotation = 0 # Degrees

    def rotate(self, angle):
        """ Rotate the object. For polygons, rotate points. For geometric shapes, update angle. """
        if self.type in ['circle', 'rect', 'square', 'heart']:
            self.rotation += angle
        elif self.type in ['triangle', 'free']:
            # For points-based shapes, we rotate the points destructively around the centroid
            pts = np.array(self.params)
            centroid = np.mean(pts, axis=0)
            
            # Rotation matrix
            rad = math.radians(angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            
            new_pts = []
            for p in pts:
                # Center point
                px, py = p[0] - centroid[0], p[1] - centroid[1]
                # Rotate
                nx = px * cos_a - py * sin_a
                ny = px * sin_a + py * cos_a
                # Translate back
                new_pts.append((int(nx + centroid[0]), int(ny + centroid[1])))
            self.params = new_pts

        def draw(self, img):
            overlay = img.copy()
            alpha = 0.4 

            if self.type == 'circle':
                center, radius = self.params
                cv2.circle(overlay, center, int(radius), self.color, cv2.FILLED)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                cv2.circle(img, center, int(radius), self.color, self.thickness)
                # Add an orientation line so we can see rotation on circles
                ex = int(center[0] + radius * math.cos(math.radians(self.rotation)))
                ey = int(center[1] + radius * math.sin(math.radians(self.rotation)))
                cv2.line(img, center, (ex, ey), (0,0,0), 2)

            elif self.type == 'rect':
                x, y, w, h = self.params
                cx, cy = x + w//2, y + h//2
            
            # Generate corners relative to center
                corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
            
            # Rotate
                rad = math.radians(self.rotation)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Apply rotation and translate back
                rotated_corners = np.dot(corners, rot_matrix.T) + [cx, cy]
                pts = rotated_corners.astype(np.int32).reshape((-1, 1, 2))

                cv2.fillPoly(overlay, [pts], self.color)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                cv2.polylines(img, [pts], True, self.color, self.thickness)

            elif self.type == 'square':
                x, y, s = self.params
                cx, cy = x + s//2, y + s//2
            
                corners = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]])
                rad = math.radians(self.rotation)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
                rotated_corners = np.dot(corners, rot_matrix.T) + [cx, cy]
                pts = rotated_corners.astype(np.int32).reshape((-1, 1, 2))

                cv2.fillPoly(overlay, [pts], self.color)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                cv2.polylines(img, [pts], True, self.color, self.thickness)

            elif self.type == 'triangle':
                # Points are already rotated in self.params
                pts = np.array(self.params, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], self.color)
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                cv2.polylines(img, [pts], True, self.color, self.thickness)

            elif self.type == 'heart':
                center, size = self.params
                t = np.linspace(0, 2 * np.pi, 50)
                x = 16 * np.sin(t)**3
                y = -(13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) 
                scale = size / 16.0