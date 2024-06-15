import cv2
import mediapipe as mp
import math
import numpy as np
import tkinter as tk
import threading
import queue

# HandMotionDetector class
class HandMotionDetector:
    def __init__(self, update_rotation_callback, positions_queue, width=1280, height=720):
        self.width = width
        self.height = height
        self.update_rotation_callback = update_rotation_callback
        self.positions_queue = positions_queue

        # Initialize MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils
        
        # Open the webcam.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    cx, cy = int(index_tip.x * self.width), int(index_tip.y * self.height)

                    self.positions_queue.put((cx, cy))

                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                    self.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Webcam Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

# RotationGUI class
class RotationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rotatable Shape")
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.pack()

        self.shape = self.canvas.create_polygon(self.calculate_initial_coords(), fill="blue")
        self.angle = 0

        self.start_webcam_button = tk.Button(self.root, text="Start Webcam", command=self.start_webcam)
        self.start_webcam_button.pack()

    def calculate_initial_coords(self):
        cx, cy = 250, 250  # Center of the canvas/shape
        w, h = 100, 150  # Width and height of the rectangle

        # Initial corner positions relative to the center with one edge turned into a triangle
        points = [
            (cx + w / 2, cy + h / 2),  # top right
            (cx, cy + h / 2 + 50),     # triangle bottom
            (cx - w / 2, cy + h / 2),  # top left
            (cx - w / 2, cy - h / 2),  # bottom left
            (cx + w / 2, cy - h / 2)   # bottom right
        ]

        return points

    def update_rotation(self, angle):
        self.angle += angle
        self.draw_rotated_shape()

    def reset_rotation(self):
        self.angle = 0
        self.draw_rotated_shape()

    def draw_rotated_shape(self):
        cx, cy = 250, 250  # Center of the canvas/shape
        w, h = 100, 150  # Width and height of the rectangle
        angle_rad = math.radians(self.angle)

        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        points = [
            (cx + w / 2 * cos_val - h / 2 * sin_val, cy + w / 2 * sin_val + h / 2 * cos_val),  # top right
            (cx + 0 * cos_val - (h / 2 + 50) * sin_val, cy + 0 * sin_val + (h / 2 + 50) * cos_val),  # triangle bottom
            (cx - w / 2 * cos_val - h / 2 * sin_val, cy - w / 2 * sin_val + h / 2 * cos_val),  # top left
            (cx - w / 2 * cos_val + h / 2 * sin_val, cy - w / 2 * sin_val - h / 2 * cos_val),  # bottom left
            (cx + w / 2 * cos_val + h / 2 * sin_val, cy + w / 2 * sin_val - h / 2 * cos_val)   # bottom right
        ]

        self.canvas.coords(self.shape, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], points[3][0], points[3][1], points[4][0], points[4][1])

    def start_webcam(self):
        self.positions_queue = queue.Queue()
        self.detector = HandMotionDetector(self.update_rotation, self.positions_queue)
        self.detector_thread = threading.Thread(target=self.detector.run)
        self.detector_thread.start()

        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.start()

    def process_queue(self):
        previous_angle = None
        while True:
            try:
                p1 = self.positions_queue.get(timeout=1)
                p2 = self.positions_queue.get(timeout=1)

                angle_p1 = math.atan2(p1[1] - 250, p1[0] - 250)
                angle_p2 = math.atan2(p2[1] - 250, p2[0] - 250)
                angle_diff = math.degrees(angle_p2 - angle_p1)

                if previous_angle is not None:
                    delta_angle = angle_diff
                    self.update_rotation(delta_angle)

                previous_angle = angle_p2

            except queue.Empty:
                continue

    def run(self):
        self.root.mainloop()

# Main script
if __name__ == "__main__":
    gui = RotationGUI()
    gui.run()
