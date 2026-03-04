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
            
            # Apply rotation to the generated heart points
            rad = math.radians(self.rotation)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            
            # Rotation matrix manual application
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            # Translate
            x_final = x_rot * scale + center[0]
            y_final = y_rot * scale + center[1]
            
            pts = np.stack((x_final, y_final), axis=1).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], self.color)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            cv2.polylines(img, [pts], True, self.color, self.thickness)
            
        elif self.type == 'free':
            points = self.params
            if len(points) > 1:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, self.color, self.thickness)

    def is_touching(self, x, y):
        margin = 30
        # For simplicity in rotation, we use approximate bounding distance checks
        # or simplified polygon tests where appropriate
        if self.type == 'circle':
            center, radius = self.params
            dist = math.hypot(x - center[0], y - center[1])
            return dist < (radius + margin)
        elif self.type == 'rect':
            rx, ry, rw, rh = self.params
            cx, cy = rx + rw//2, ry + rh//2
            # Approximation: Distance check to center vs max radius
            # For accurate hit test on rotated rect, we should inverse rotate point
            # but simple radius check is usually "good enough" for UX
            dist = math.hypot(x - cx, y - cy)
            return dist < (max(rw, rh)/2 + margin)
        elif self.type == 'square':
            rx, ry, s = self.params
            cx, cy = rx + s//2, ry + s//2
            dist = math.hypot(x - cx, y - cy)
            return dist < (s/2 * 1.4 + margin) # Diagonal approx
        elif self.type == 'triangle':
            pts = np.array(self.params, np.int32)
            dist = cv2.pointPolygonTest(pts, (x, y), True)
            return dist > -margin
        elif self.type == 'heart':
            center, size = self.params
            dist = math.hypot(x - center[0], y - center[1])
            return dist < (size + margin)
        elif self.type == 'free':
            pts = np.array(self.params)
            if len(pts) == 0: return False
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            return (x_min - margin < x < x_max + margin) and (y_min - margin < y < y_max + margin)
        return False

    def move(self, dx, dy):
        if self.type == 'circle' or self.type == 'heart':
            cx, cy = self.params[0]
            self.params[0] = (cx + dx, cy + dy)
        elif self.type == 'rect':
            self.params[0] += dx
            self.params[1] += dy
        elif self.type == 'square':
            self.params[0] += dx
            self.params[1] += dy
        elif self.type == 'triangle':
            new_pts = []
            for px, py in self.params:
                new_pts.append((px + dx, py + dy))
            self.params = new_pts
        elif self.type == 'free':
            new_pts = []
            for px, py in self.params:
                new_pts.append((px + dx, py + dy))
            self.params = new_pts

    def scale(self, factor):
        if factor <= 0: return
        # Limit to reasonable sizes to prevent crashes or invisible elements
        min_size = 20
        max_size = 2000
        
        if self.type == 'circle' or self.type == 'heart':
            new_r = int(self.params[1] * factor)
            self.params[1] = max(min_size, min(max_size, new_r))
        elif self.type == 'rect':
            x, y, w, h = self.params
            cx, cy = x + w//2, y + h//2
            new_w, new_h = int(w * factor), int(h * factor)
            if new_w < min_size or new_h < min_size or new_w > max_size or new_h > max_size: return
            self.params = [cx - new_w//2, cy - new_h//2, new_w, new_h]
        elif self.type == 'square':
            x, y, s = self.params
            cx, cy = x + s//2, y + s//2
            new_s = max(min_size, min(max_size, int(s * factor)))
            self.params = [cx - new_s//2, cy - new_s//2, new_s]
        elif self.type == 'triangle':
            pts = np.array(self.params)
            centroid = np.mean(pts, axis=0)
            new_pts = []
            
            # test size by checking distance of first point to centroid
            if len(pts) > 0:
                 current_dist = math.hypot(pts[0][0] - centroid[0], pts[0][1] - centroid[1])
                 new_dist = current_dist * factor
                 if new_dist < min_size or new_dist > max_size: return 

            for p in pts:
                vec = p - centroid
                new_p = centroid + vec * factor
                new_pts.append((int(new_p[0]), int(new_p[1])))
            self.params = new_pts

def smooth_points(points):
    """ Smooths the drawn stroke to remove hand jitter """
    if len(points) < 5: return points
    pts = np.array(points)
    kernel = np.ones(5) / 5
    x_smooth = np.convolve(pts[:, 0], kernel, mode='valid')
    y_smooth = np.convolve(pts[:, 1], kernel, mode='valid')
    smoothed = np.stack([x_smooth, y_smooth], axis=1).astype(np.int32)
    return [tuple(p) for p in smoothed]

def detect_shape(points):
    """ Uses the Trained ML Model to predict the shape with HYBRID VALIDATION """
    if len(points) < 20: return None
    points = smooth_points(points)
    if len(points) < 10: return None

    contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    
    # 1. Close the loop check
    peri_open = cv2.arcLength(contour, False)
    start_point = points[0]
    end_point = points[-1]
    dist_open = math.hypot(start_point[0]-end_point[0], start_point[1]-end_point[1])
    is_closed = dist_open < max(100, 0.40 * peri_open) if peri_open > 0 else False
    
    if not is_closed:
        return ShapeObject('free', points, (255, 0, 255)) 

    # Close the contour
    contour_closed = np.vstack([contour, contour[0:1]])
    
    # Calculate geometric stats for SANITY CHECK
    area = cv2.contourArea(contour_closed)
    peri = cv2.arcLength(contour_closed, True)
    if peri == 0: return ShapeObject('free', points, (255, 0, 255))
    
    circularity = 4 * math.pi * area / (peri * peri)
    hull = cv2.convexHull(contour_closed)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    
    x, y, w, h = cv2.boundingRect(contour_closed)
    center = (x + w//2, y + h//2)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    pts_squeeze = contour_closed.squeeze()
    if pts_squeeze.ndim == 1: pts_squeeze = pts_squeeze.reshape(-1, 2)
    
    # Calculate radius variation to detect circles securely
    if len(pts_squeeze) > 0:
         dists = np.sqrt(np.sum((pts_squeeze - center)**2, axis=1))
         radius_mean = np.mean(dists)
         radius_std = np.std(dists)
         radius_cv = radius_std / radius_mean if radius_mean > 0 else 0
    else:
         radius_mean, radius_cv = 0, 0

    # === ML PREDICTION ===
    label, confidence = clf.predict(contour_closed)
    
    if confidence > 8000:
         return ShapeObject('free', points, (255, 0, 255))
    
    # === HYBRID VALIDATION ===
    if radius_cv < 0.14:
        return ShapeObject('circle', [center, int(radius_mean)], (0, 255, 255))
    
    if label == 0: # Circle
        if extent > 0.90:
             size = max(w, h)
             return ShapeObject('square', [x, y, size], (255, 100, 0))
        if circularity > 0.65 or radius_cv < 0.20: 
            return ShapeObject('circle', [center, int(radius_mean)], (0, 255, 255))
        
    elif label == 1: # Triangle
        if circularity < 0.75:
             hull = cv2.convexHull(contour_closed)
             approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
             if len(approx) == 3:
                 pts = [tuple(p[0]) for p in approx]
                 return ShapeObject('triangle', pts, (0, 255, 0))
             else:
                pt1 = (center[0], y)
                pt2 = (x, y+h)
                pt3 = (x+w, y+h)
                return ShapeObject('triangle', [pt1, pt2, pt3], (0, 255, 0))

    elif label == 2: # Quad
        if extent < 0.85 and circularity > 0.82:
             return ShapeObject('circle', [center, int(radius_mean)], (0, 255, 255))
        if solidity > 0.7:
            aspect_ratio = float(w)/float(h)
            if 0.75 <= aspect_ratio <= 1.3:
                size = max(w, h)
                return ShapeObject('square', [x, y, size], (255, 100, 0))
            else:
                return ShapeObject('rect', [x, y, w, h], (0, 255, 0))

    elif label == 3: # Heart
        if circularity < 0.9:
            size = max(w, h) // 2
            return ShapeObject('heart', [center, size], (203, 192, 255))

    return ShapeObject('free', points, (255, 0, 255))

def main():
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        print("Please check your camera connections or permissions.")
        return

    cap.set(3, 1280)
    cap.set(4, 720)
    detector = HandDetector(maxHands=2)
    
    objects = [] 
    current_stroke = [] 
    
    active_object = None
    last_pinch_pos = None
    last_hand_dist = None
    last_angle = None # Track rotation angle
    
    print("Futuristic Canvas Loaded.")
    print("ML Model Enabled (Robust Hybrid Mode + Enhanced Circle Check).")
    print("Draw: Circles, Rects, Squares, Triangles, Hearts")
    print("Manipulation: Pinch + Two Hands to Expand AND Rotate")
    
    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        img = cv2.addWeighted(img, 0.7, np.zeros(img.shape, img.dtype), 0.3, 0)
        
        img = detector.findHands(img, draw=True)
        lmList1 = detector.findPosition(img, handNo=0)
        lmList2 = detector.findPosition(img, handNo=1)

        right_hand_lms = []
        left_hand_lms = []
        if lmList1:
            if not lmList2: right_hand_lms = lmList1
            elif lmList1[0][1] > lmList2[0][1]: 
                right_hand_lms = lmList1
                left_hand_lms = lmList2
            else: 
                right_hand_lms = lmList2
                left_hand_lms = lmList1

        for obj in objects:
            obj.draw(img)

        if right_hand_lms:
            x1, y1 = right_hand_lms[8][1:]   # Index
            x2, y2 = right_hand_lms[12][1:]  # Middle
            x_thumb, y_thumb = right_hand_lms[4][1:] # Thumb
            
            fingers = detector.fingersUp(right_hand_lms)
            pinch_dist = math.hypot(x1 - x_thumb, y1 - y_thumb)
            is_pinching = pinch_dist < 40
            
            is_drawing_gesture = fingers[1] and not fingers[2] and not is_pinching
            is_eraser_gesture = all(fingers) 

            if is_drawing_gesture:
                active_object = None 
                current_stroke.append((x1, y1))
                if len(current_stroke) > 1:
                    pts = np.array(current_stroke, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], False, (255, 255, 255), 2)
                cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)

            else:
                if len(current_stroke) > 0:
                    new_shape = detect_shape(current_stroke)
                    if new_shape:
                        objects.append(new_shape)
                    current_stroke = [] 

                if is_eraser_gesture:
                    active_object = None
                    eraser_pos = (x2, y2) 
                    cv2.circle(img, eraser_pos, 25, (0, 0, 0), cv2.FILLED) 
                    cv2.circle(img, eraser_pos, 25, (0, 0, 255), 2)
                    cv2.putText(img, "ERASER", (eraser_pos[0]-30, eraser_pos[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    
                    for obj in reversed(objects):
                        if obj.is_touching(eraser_pos[0], eraser_pos[1]):
                            objects.remove(obj)

                elif is_pinching:
                    pinch_pos = ((x1+x_thumb)//2, (y1+y_thumb)//2)
                    cv2.circle(img, pinch_pos, 10, (0, 255, 0), cv2.FILLED)
                    
                    if active_object is None:
                        for obj in reversed(objects): 
                            if obj.is_touching(pinch_pos[0], pinch_pos[1]):
                                active_object = obj
                                last_pinch_pos = pinch_pos
                                break
                    
                    if active_object:
                        cv2.putText(img, "LOCKED", (pinch_pos[0]+20, pinch_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        
                        if last_pinch_pos:
                            dx = pinch_pos[0] - last_pinch_pos[0]
                            dy = pinch_pos[1] - last_pinch_pos[1]
                            active_object.move(dx, dy)
                        last_pinch_pos = pinch_pos
                        
                        if left_hand_lms:
                            lx, ly = left_hand_lms[8][1:]
                            cv2.line(img, pinch_pos, (lx, ly), (255, 255, 0), 2)
                            
                            # 1. SCALING LOGIC
                            curr_hand_dist = math.hypot(pinch_pos[0]-lx, pinch_pos[1]-ly)
                            if last_hand_dist:
                                scale_factor = curr_hand_dist / last_hand_dist
                                if abs(scale_factor - 1) > 0.01:
                                    active_object.scale(scale_factor)
                            last_hand_dist = curr_hand_dist
                            
                            # 2. ROTATION LOGIC
                            # Calculate angle of the line connecting two hands relative to horizontal
                            curr_angle = math.degrees(math.atan2(ly - pinch_pos[1], lx - pinch_pos[0]))
                            if last_angle is not None:
                                delta_angle = curr_angle - last_angle
                                # Handle wrap around (e.g. 179 to -179)
                                # Simple fix for smooth rotation without wrapping logic for now:
                                active_object.rotate(delta_angle)
                            last_angle = curr_angle
                            
                        else:
                            last_hand_dist = None
                            last_angle = None

                else:
                    active_object = None
                    last_pinch_pos = None
                    last_hand_dist = None
                    last_angle = None
                
        cv2.putText(img, f"Objects: {len(objects)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        try:
           cv2.imshow("Futuristic Gesture Canvas", img)
        except Exception as e:
           print(f"Display Error: {e}")
           break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()