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

