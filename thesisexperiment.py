import tkinter as tk
import cv2
import mediapipe as mp
import threading
import math
import time
import random
import os

# Global variables for hold detection
target_angle = 0
holding_start_time = None
holding_duration = 3  # seconds
current_angle = 0
trial_count = 0
block_count = 0
technique_count = 0
trials_per_block = 30
blocks_per_technique = 4
total_techniques = 8
total_blocks = blocks_per_technique * total_techniques
target_radius = 300  # Radius from the center where the targets will appear

# Data storage
trial_data = []
participant_id = None
current_rotation_type = 1  # Initialize with a valid rotation type
technique_sequence = []  # Will be set based on participant_id
previous_trial_data = None  # Variable to store the previous trial's data

# Example of rotational techniques dictionary
rotation_techniques = {
    1: "index_finger_tip",
    2: "thumb_tip",
    3: "middle_finger_tip",
    4: "ring_finger_tip",
    5: "pinky_tip",
    6: "all_fingers",
    7: "wrist",
    8: "arm"
}

# Define hardcoded target positions for 4 blocks
block_targets = {
    1: [
        (300, 100), (500, 100), (300, 500), (500, 500),
        (250, 150), (550, 150), (250, 450), (550, 450),
        (200, 200), (600, 200), (200, 400), (600, 400),
        (150, 250), (650, 250), (150, 350), (650, 350),
        (100, 300), (700, 300), (100, 500), (700, 500),
        (50, 350), (750, 350), (50, 450), (750, 450),
        (150, 150), (650, 150), (150, 550), (650, 550),
        (200, 100), (600, 100)
    ],
    2: [
        (750, 450), (150, 150), (600, 100), (250, 150),
        (500, 100), (650, 350), (50, 350), (650, 250),
        (300, 500), (150, 250), (550, 150), (700, 300),
        (300, 100), (600, 200), (700, 500), (50, 450),
        (550, 450), (200, 100), (250, 450), (600, 400),
        (100, 300), (500, 500), (150, 350), (700, 500),
        (200, 200), (200, 400), (100, 500), (150, 550),
        (650, 150), (750, 350)
    ],
    3: [
        (200, 100), (500, 500), (650, 350), (300, 100),
        (250, 450), (100, 300), (200, 200), (750, 450),
        (150, 150), (650, 250), (500, 100), (200, 400),
        (50, 350), (700, 500), (550, 150), (600, 100),
        (600, 400), (750, 350), (150, 350), (300, 500),
        (700, 300), (50, 450), (600, 200), (150, 250),
        (150, 550), (550, 450), (650, 150), (200, 400),
        (700, 500), (200, 100)
    ],
    4: [
        (550, 150), (750, 350), (300, 100), (150, 150),
        (650, 150), (250, 450), (200, 400), (150, 250),
        (600, 400), (50, 450), (700, 500), (100, 300),
        (300, 500), (500, 500), (700, 300), (600, 100),
        (200, 100), (700, 500), (250, 150), (650, 250),
        (600, 200), (50, 350), (500, 100), (150, 350),
        (650, 350), (150, 550), (550, 450), (100, 500),
        (750, 450), (200, 200)
    ]
}


# Example technique sequences for each user ID
technique_sequences = {
    1: [1, 2, 3, 4, 5, 6, 7, 8],
    2: [2, 3, 4, 5, 6, 7, 8, 1],
    3: [3, 4, 5, 6, 7, 8, 1, 2],
    4: [4, 5, 6, 7, 8, 1, 2, 3],
    5: [5, 6, 7, 8, 1, 2, 3, 4],
    6: [6, 7, 8, 1, 2, 3, 4, 5],
    7: [7, 8, 1, 2, 3, 4, 5, 6],
    8: [8, 1, 2, 3, 4, 5, 6, 7],
    9: [1, 3, 5, 7, 2, 4, 6, 8],
    10: [2, 4, 6, 8, 1, 3, 5, 7],
    11: [3, 5, 7, 1, 4, 6, 8, 2],
    12: [4, 6, 8, 2, 5, 7, 1, 3]
}

# Function to rotate rectangles
def rotate_rectangle(x, y, landmark):
    global current_angle
    current_angle = math.atan2(y - canvas_height / 2, x - canvas_width / 2)
    offset_width = 50  # half of rectangle's width
    offset_length = 100  # half of rectangle's length

    # Calculate coordinates for the red half
    red_coords = [
        canvas_width / 2 + math.cos(current_angle) * offset_length, canvas_height / 2 + math.sin(current_angle) * offset_length,
        canvas_width / 2 + math.cos(current_angle + math.pi / 2) * offset_width, canvas_height / 2 + math.sin(current_angle + math.pi / 2) * offset_width,
        canvas_width / 2, canvas_height / 2,
        canvas_width / 2 + math.cos(current_angle - math.pi / 2) * offset_width, canvas_height / 2 + math.sin(current_angle - math.pi / 2) * offset_width
    ]

    # Calculate coordinates for the green half
    green_coords = [
        canvas_width / 2, canvas_height / 2,
        canvas_width / 2 + math.cos(current_angle + math.pi / 2) * offset_width, canvas_height / 2 + math.sin(current_angle + math.pi / 2) * offset_width,
        canvas_width / 2 + math.cos(current_angle + math.pi) * offset_length, canvas_height / 2 + math.sin(current_angle + math.pi) * offset_length,
        canvas_width / 2 + math.cos(current_angle - math.pi / 2) * offset_width, canvas_height / 2 + math.sin(current_angle - math.pi / 2) * offset_width
    ]

    canvas.coords(red_rectangle, *red_coords)
    canvas.coords(green_rectangle, *green_coords)

# Function to generate a random target position on the circumference
def generate_target():
    angle = random.uniform(0, 2 * math.pi)
    x = canvas_width / 2 + target_radius * math.cos(angle)
    y = canvas_height / 2 + target_radius * math.sin(angle)
    return int(x), int(y)

# Function to draw the target on the canvas
def draw_target(x, y):
    target_radius = 10
    canvas.create_oval(x - target_radius, y - target_radius, x + target_radius, y + target_radius, fill='red', tags="target")
    canvas.create_line(canvas_width / 2, canvas_height / 2, x, y, fill='green', tags="target", width=3)  # Set the line width to 3 pixels

# Function to show trial complete message
def show_trial_complete_message():
    canvas.itemconfig(trial_message, text="Trial Complete!")
    root.after(2000, lambda: canvas.itemconfig(trial_message, text=""))  # Clear message after 2 seconds

# Function to update trial label
def update_trial_label():
    trial_label.config(text=f"Trial {trial_count + 1} out of {trials_per_block}")
    block_label.config(text=f"Block {block_count + 1} out of {total_blocks}")
    rotation_label.config(text=f"Rotation Type: {rotation_techniques[current_rotation_type]}")

# Function to clear the canvas
def clear_canvas():
    canvas.delete("target")  # Delete all items with the tag "target"

# Function to write data to a file
def write_data_to_file():
    print("Writing data to file...")
    global block_count
    filename = f"{participant_id}{rotation_techniques[current_rotation_type]}_block{block_count}.txt"
    filepath = os.path.abspath(filename)
    print(f"Saving data to: {filepath}")  # Debug print statement for the file path
    with open(filepath, "w") as file:
        file.write("Trial,Duration(ms),Target Angle,Current Angle\n")
        for trial in trial_data:
            file.write(f"{trial['trial']},{trial['duration']},{trial['target_angle']},{trial['current_angle']}\n")
    print(f"Data written to {filepath}")
    trial_data.clear()  # Clear the trial data for the next block

# Function to start the next block
def start_next_block():
    global block_count, technique_count, current_rotation_type, trial_count
    trial_count = 0
    block_count += 1
    print(f"Starting next block: {block_count}")
    if block_count <= total_blocks:
        if block_count % blocks_per_technique == 0:
            technique_count += 1
            current_rotation_type = technique_sequence[technique_count % total_techniques]
        start_next_block_button.pack_forget()  # Hide the button before starting the trials
        initialize_webcam_and_mediapipe()  # Reinitialize resources for the new block
        start_trials()
    else:
        canvas.itemconfig(trial_message, text="All trials complete!")

# Function to initialize webcam and MediaPipe Hands
def initialize_webcam_and_mediapipe():
    global cap, hands
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    print("Camera successfully opened.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to close webcam and MediaPipe Hands
def close_webcam_and_mediapipe():
    print("Closing webcam and MediaPipe resources...")
    global cap, hands
    if cap is not None and cap.isOpened():
        print("Releasing camera.")
        cap.release()
    if hands is not None:
        print("Closing MediaPipe Hands.")
        hands.close()
    #cv2.destroyAllWindows()

# Function to start the next trial
def start_next_trial():
    threading.Thread(target=next_trial).start()  # Start the next trial in a separate thread

def next_trial():
    global holding_start_time, target_angle, trial_count, block_count, technique_count, current_rotation_type, trial_start_time

    if trial_count >= trials_per_block:
        print(f"Trials per block reached: {trial_count}")
        close_webcam_and_mediapipe()  # Close the camera and MediaPipe resources after each block
        write_data_to_file()
        start_next_block_button.pack(side=tk.LEFT, padx=10)  # Show the button to start the next block
        return

    clear_canvas()  # Clear previous target

    # Use the predefined target positions
    block_index = (block_count % 4) + 1
    target_x, target_y = block_targets[block_index][trial_count % len(block_targets[block_index])]

    draw_target(target_x, target_y)
    target_angle = math.atan2(target_y - canvas_height / 2, target_x - canvas_width / 2)
    update_trial_label()
    holding_start_time = None  # Reset holding start time
    trial_start_time = time.time()  # Record the start time of the trial

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_name = rotation_techniques[current_rotation_type]
                    
                    if landmark_name == "all_fingers" or landmark_name == "arm":
                        indices = [4, 8, 12, 16, 20]
                        x = int(sum(hand_landmarks.landmark[i].x for i in indices) / len(indices) * frame.shape[1])
                        y = int(sum(hand_landmarks.landmark[i].y for i in indices) / len(indices) * frame.shape[0])
                    elif landmark_name == "wrist":
                        indices = [0, 5, 9, 13, 17]
                        x = int(sum(hand_landmarks.landmark[i].x for i in indices) / len(indices) * frame.shape[1])
                        y = int(sum(hand_landmarks.landmark[i].y for i in indices) / len(indices) * frame.shape[0])
                    else:
                        landmark_index = getattr(mp.solutions.hands.HandLandmark, landmark_name.upper())
                        landmark_coords = hand_landmarks.landmark[landmark_index]
                        x = int(landmark_coords.x * frame.shape[1])
                        y = int(landmark_coords.y * frame.shape[0])
                    
                    rotate_rectangle(x, y, landmark_name)
                    print(f"Landmark {landmark_name} Position: ({x}, {y}), Current Angle: {current_angle}, Target Angle: {target_angle}")

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Webcam Feed', frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

def end_trial(event=None):  # Add default argument to handle event
    global trial_start_time, trial_count

    trial_end_time = time.time()
    duration = int((trial_end_time - trial_start_time) * 1000)  # Duration in milliseconds
    final_current_angle = current_angle
    print("Manual trial end. Trial complete.")
    trial_count += 1
    trial_data.append({
        "trial": trial_count,
        "duration": duration,
        "target_angle": target_angle,
        "current_angle": final_current_angle
    })
    store_previous_trial_data()  # Store the previous trial data
    show_trial_complete_message()
    if trial_count < trials_per_block:
        next_trial_button.pack(side=tk.LEFT, padx=10)  # Ensure the button to start the next trial is visible
    else:
        print("Completed all trials in the block. Preparing to save data and start the next block.")
        close_webcam_and_mediapipe()  # Close the camera and MediaPipe resources after the block ends
        write_data_to_file()  # Ensure this is called to save the data
        start_next_block_button.pack(side=tk.LEFT, padx=10)  # Show the button to start the next block

    redo_trial_button.pack(side=tk.LEFT, padx=10)  # Show the redo button

def start_trials():
    start_button.pack_forget()  # Hide the start button after trials start
    start_next_trial()  # Start the first trial

def get_participant_id():
    global participant_id, current_rotation_type, technique_sequence
    participant_id = int(participant_id_entry.get())
    if participant_id and 1 <= participant_id <= 12:
        participant_id_window.destroy()
        technique_sequence = technique_sequences[participant_id]
        current_rotation_type = technique_sequence[0]  # Initialize with the first technique in the sequence
        root.deiconify()
        update_trial_label()  # Update labels with initial values

def store_previous_trial_data():
    global previous_trial_data
    previous_trial_data = {
        "trial": trial_count,
        "duration": int((time.time() - trial_start_time) * 1000),  # Duration in milliseconds
        "target_angle": target_angle,
        "current_angle": current_angle
    }

def redo_previous_trial():
    global trial_count, trial_data, previous_trial_data, target_angle, current_angle, trial_start_time

    if previous_trial_data:
        # Decrement trial count to redo the previous trial
        trial_count -= 1

        # Remove the last trial data entry if it exists
        if trial_data:
            trial_data.pop()

        # Reset angles and start time
        target_angle = previous_trial_data["target_angle"]
        current_angle = previous_trial_data["current_angle"]
        trial_start_time = time.time()

        # Redraw target and update trial label
        clear_canvas()
        draw_target(int(canvas_width / 2 + target_radius * math.cos(target_angle)),
                    int(canvas_height / 2 + target_radius * math.sin(target_angle)))
        update_trial_label()
        print("Redoing previous trial...")

        # Hide redo button until the trial is completed again
        redo_trial_button.pack_forget()

        # Start the trial again
        threading.Thread(target=next_trial).start()

# Create main window
root = tk.Tk()
root.withdraw()  # Hide main window initially
root.title("Rotation Experiment")
canvas_width = 800
canvas_height = 600

# Create canvas
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Draw rectangles
red_rectangle = canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0, fill='green')
green_rectangle = canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0, fill='red')

# Create trial complete message (initially empty)
trial_message = canvas.create_text(canvas_width / 2, canvas_height - 20, text="", font=("Helvetica", 16), fill="red")

# Create trial label
trial_label = tk.Label(root, text="", font=("Helvetica", 16))
trial_label.pack(pady=10)

# Create block label
block_label = tk.Label(root, text="", font=("Helvetica", 16))
block_label.pack(pady=10)

# Create rotation type label
rotation_label = tk.Label(root, text="", font=("Helvetica", 16))
rotation_label.pack(pady=10)

# Create start button
start_button = tk.Button(root, text="Start Trials", command=start_trials)
start_button.pack(pady=10)  # Adjust padding to ensure it's within bounds

# Create start next trial button (initially visible and placed side by side)
next_trial_button = tk.Button(root, text="Start Next Trial", command=start_next_trial)
next_trial_button.pack(side=tk.LEFT, padx=10)

# Create end trial button
end_trial_button = tk.Button(root, text="End Trial", command=end_trial)
end_trial_button.pack(side=tk.LEFT, padx=10)

# Create start next block button (initially hidden)
start_next_block_button = tk.Button(root, text="Start Next Block", command=start_next_block)
start_next_block_button.pack_forget()

# Create redo previous trial button (initially hidden)
redo_trial_button = tk.Button(root, text="Redo Previous Trial", command=redo_previous_trial)
redo_trial_button.pack_forget()

# Create participant ID input window
participant_id_window = tk.Toplevel(root)
participant_id_window.title("Enter Participant ID")
tk.Label(participant_id_window, text="Participant ID:").pack(pady=10)
participant_id_entry = tk.Entry(participant_id_window)
participant_id_entry.pack(pady=10)
tk.Button(participant_id_window, text="Submit", command=get_participant_id).pack(pady=10)

# Bind spacebar key to end_trial function
root.bind('<space>', end_trial)

# Initialize webcam and MediaPipe Hands before starting
initialize_webcam_and_mediapipe()

# Run the main loop
root.mainloop()
