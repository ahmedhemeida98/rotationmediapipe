import cv2
import mediapipe as mp
import time
import math
import numpy as np
from scipy.optimize import leastsq
import tkinter as tk
import threading

# HandMotionDetector class
class HandMotionDetector:
    def __init__(self, update_rotation_callback, width=1280, height=720, motion_threshold=20, interval=0.2):
        self.width = width
        self.height = height
        self.motion_threshold = motion_threshold
        self.interval = interval
        self.positions = []
        self.timestamps = []
        self.path = []  # List to store the path of the hand movement
        self.tracking = False  # Variable to control tracking state
        self.last_angle = None
        self.total_rotation = 0
        self.update_rotation_callback = update_rotation_callback

        # Initialize MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils
        
        # Open the webcam.
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.width)
        self.cap.set(4, self.height)

    def append_values(self, cx, cy, current_time):
        self.positions.append((cx, cy))
        self.timestamps.append(current_time)

        # Remove old data outside the interval
        while self.timestamps and current_time - self.timestamps[0] > self.interval:
            self.positions.pop(0)
            self.timestamps.pop(0)

    def compute_distance(self):
        step_distances = []
        for i in range(1, len(self.positions)):
            previous_pos = self.positions[i - 1]
            current_pos = self.positions[i]
            step_distance = math.sqrt((current_pos[0] - previous_pos[0])**2 + (current_pos[1] - previous_pos[1])**2)
            step_distances.append(step_distance)
        return sum(step_distances)

    def fit_circle(self, x, y):
        def calc_R(xc, yc):
            return np.sqrt((x - xc)**2 + (y - yc)**2)

        def f_2(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = np.mean(x), np.mean(y)
        center, _ = leastsq(f_2, center_estimate)
        xc, yc = center
        Ri = calc_R(*center)
        R = Ri.mean()
        return xc, yc, R

    def compute_angles(self, cx, cy):
        angles = [np.arctan2(y - cy, x - cx) for x, y in self.path]
        return angles

    def compute_rotation(self, angles):
        total_rotation = 0
        for i in range(1, len(angles)):
            delta_angle = angles[i] - angles[i - 1]
            if delta_angle > np.pi:
                delta_angle -= 2 * np.pi
            elif delta_angle < -np.pi:
                delta_angle += 2 * np.pi
            total_rotation += delta_angle
        return total_rotation

    def start_tracking(self):
        self.tracking = True
        self.path = []  # Clear any existing path data
        self.total_rotation = 0
        self.last_angle = None

    def stop_tracking(self):
        self.tracking = False
        if self.path:
            x = np.array([p[0] for p in self.path])
            y = np.array([p[1] for p in self.path])
            cx, cy, r = self.fit_circle(x, y)
            angles = self.compute_angles(cx, cy)
            total_rotation = self.compute_rotation(angles)

            print(f"Circle center: ({cx}, {cy}), Radius: {r}")
            print(f"Total Rotation: {np.degrees(total_rotation):.2f} degrees")

            # Determine if the path forms part of a circle or a complete circle
            path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            circumference = 2 * np.pi * r
            if path_length >= circumference:
                circle_type = "Complete Circle"
            else:
                circle_type = "Part of a Circle"

            print(f"Path Type: {circle_type}")

            # Store results for visualization
            self.circle_center = (int(cx), int(cy))
            self.circle_radius = int(r)
            self.total_rotation = np.degrees(total_rotation)
            self.circle_type = circle_type

            # Update rotation in the GUI
            self.update_rotation_callback(self.total_rotation)

    def delete_path(self):
        self.path = []

    def append_to_path(self, cx, cy):
        if self.tracking:
            self.path.append((cx, cy))

    def visualize_path(self, frame):
        for i in range(1, len(self.path)):
            cv2.line(frame, self.path[i - 1], self.path[i], (0, 255, 0), 2)

    def visualize(self, frame, hand_landmarks, motion_detected):
        for hand_landmark in hand_landmarks:
            index_tip = hand_landmark.landmark[8]
            cx, cy = int(index_tip.x * self.width), int(index_tip.y * self.height)
            
            # Append values to the lists
            self.append_values(cx, cy, time.time())
            # Append to the path
            self.append_to_path(cx, cy)

            # Optionally, draw circles at the index finger tip.
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

            self.drawing_utils.draw_landmarks(
                frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

        # Display "Motion detected" text on the frame
        if motion_detected:
            cv2.putText(frame, 'Motion detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Visualize the path
        self.visualize_path(frame)

        # If tracking has stopped, visualize the circle and rotation information
        if not self.tracking and hasattr(self, 'circle_center'):
            cv2.circle(frame, self.circle_center, self.circle_radius, (255, 0, 0), 2)
            cv2.putText(frame, f'Center: {self.circle_center}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Radius: {self.circle_radius}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Total Rotation: {self.total_rotation:.2f} degrees', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Path Type: {self.circle_type}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame.
        cv2.imshow('Webcam Feed', frame)

    def run(self):
        while True:
            # Capture frame-by-frame.
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip the frame horizontally for a later selfie-view display.
            frame = cv2.flip(frame, 1)

            # Convert the BGR image to RGB before processing.
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            motion_detected = False

            # Draw the hand annotations on the image.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    cx, cy = int(index_tip.x * self.width), int(index_tip.y * self.height)

                    # Append values to the lists
                    self.append_values(cx, cy, time.time())
                    # Append to the path
                    self.append_to_path(cx, cy)

                    # Compute the total distance
                    total_distance = self.compute_distance()

                    # Check if the total distance exceeds the motion threshold
                    if total_distance > self.motion_threshold:
                        motion_detected = True
                        if not self.tracking:
                            self.start_tracking()
                        print("Motion detected")
                    else:
                        if self.tracking:
                            self.stop_tracking()

                # Visualize the results
                self.visualize(frame, results.multi_hand_landmarks, motion_detected)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.delete_path()

        # When everything is done, release the capture and destroy the window.
        self.cap.release()
        cv2.destroyAllWindows()

        # Close the MediaPipe Hands instance.
        self.hands.close()

# RotationGUI class
class RotationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Rotatable Rectangle")
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.pack()

        self.rect = self.canvas.create_polygon(self.calculate_initial_coords(), fill="blue")
        self.angle = 0

    def calculate_initial_coords(self):
        cx, cy = 250, 250  # Center of the canvas/rectangle
        w, h = 100, 150  # Width and height of the rectangle

        # Initial corner positions relative to the center
        points = [
            (cx + w / 2, cy + h / 2),  # top right
            (cx - w / 2, cy + h / 2),  # top left
            (cx - w / 2, cy - h / 2),  # bottom left
            (cx + w / 2, cy - h / 2)   # bottom right
        ]

        return points

    def update_rotation(self, angle):
        self.angle = angle
        self.draw_rotated_rectangle()

    def draw_rotated_rectangle(self):
        cx, cy = 250, 250  # Center of the canvas/rectangle
        w, h = 100, 150  # Width and height of the rectangle
        angle_rad = math.radians(self.angle)

        # Calculate the new corner positions after rotation
        cos_val = math.cos(angle_rad)
        sin_val = math.sin(angle_rad)

        points = [
            (cx + w / 2 * cos_val - h / 2 * sin_val, cy + w / 2 * sin_val + h / 2 * cos_val),  # top right
            (cx - w / 2 * cos_val - h / 2 * sin_val, cy - w / 2 * sin_val + h / 2 * cos_val),  # top left
            (cx - w / 2 * cos_val + h / 2 * sin_val, cy - w / 2 * sin_val - h / 2 * cos_val),  # bottom left
            (cx + w / 2 * cos_val + h / 2 * sin_val, cy + w / 2 * sin_val - h / 2 * cos_val)   # bottom right
        ]

        # Update the rectangle coordinates on the canvas
        self.canvas.coords(self.rect, points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], points[3][0], points[3][1])

    def run(self):
        self.root.mainloop()

# Main script
if __name__ == "__main__":
    gui = RotationGUI()
    
    # Run the hand motion detector in a separate thread
    detector = HandMotionDetector(gui.update_rotation)
    detector_thread = threading.Thread(target=detector.run)
    detector_thread.start()
    
    # Run the GUI in the main thread
    gui.run()
