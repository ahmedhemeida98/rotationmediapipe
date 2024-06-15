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
    1: [(300, 100), (500, 100), (300, 500), (500, 500)],
    2: [(200, 200), (600, 200), (200, 400), (600, 400)],
    3: [(100, 300), (700, 300), (100, 500), (700, 500)],
    4: [(150, 150), (650, 150), (150, 550), (650, 550)]
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
    canvas.create_line(canvas_width / 2, canvas_height / 2, x, y, fill='green', tags="target", width=3)  # Set the line width to 5 pixels

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
    global block_count
    filename = f"{participant_id}_{rotation_techniques[current_rotation_type]}_block_{block_count + 1}.txt"
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
    if block_count < total_blocks:
        if block_count % blocks_per_technique == 0:
            technique_count += 1
            current_rotation_type = technique_sequence[technique_count % total_techniques]
        start_next_block_button.pack_forget()  # Hide the button before starting the trials
        start_trials()
    else:
        canvas.itemconfig(trial_message, text="All trials complete!")

# Function to start the next trial
def start_next_trial():
    next_trial_button.pack_forget()  # Hide the button before starting the next trial
    threading.Thread(target=next_trial).start()  # Start the next trial in a separate thread

def next_trial():
    global holding_start_time, target_angle, trial_count, block_count, technique_count, current_rotation_type, cap, hands, trial_start_time

    if trial_count >= trials_per_block:
        print(f"Trials per block reached: {trial_count}")
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

    try:
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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
                        indices = [5, 9, 13, 17]
                        x = int(sum(hand_landmarks.landmark[i].x for i in indices) / len(indices) * frame.shape[1])
                        y = int(sum(hand_landmarks.landmark[i].y for i in indices) / len(indices) * frame.shape[0])
                    else:
                        landmark = getattr(mp.solutions.hands.HandLandmark, landmark_name.upper())
                        landmark_coords = hand_landmarks.landmark[landmark]
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
        cv2.destroyAllWindows()

# Function to manually end the trial
def end_trial():
    global trial_start_time, trial_count, cap

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
    show_trial_complete_message()
    if cap.isOpened():
        cap.release()  # Release the webcam for the next trial
    if trial_count < trials_per_block:
        next_trial_button.pack(side=tk.LEFT, padx=10)  # Ensure the button to start the next trial is visible
    else:
        print("Completed all trials in the block. Preparing to save data and start the next block.")
        write_data_to_file()  # Ensure this is called to save the data
        start_next_block_button.pack(side=tk.LEFT, padx=10)  # Show the button to start the next block

# Function to start the trials
def start_trials():
    start_button.pack_forget()  # Hide the start button after trials start
    start_next_trial()  # Start the first trial

# Function to get participant ID and set rotation sequence
def get_participant_id():
    global participant_id, current_rotation_type, technique_sequence
    participant_id = int(participant_id_entry.get())
    if participant_id and 1 <= participant_id <= 12:
        participant_id_window.destroy()
        technique_sequence = technique_sequences[participant_id]
        current_rotation_type = technique_sequence[0]  # Initialize with the first technique in the sequence
        root.deiconify()
        update_trial_label()  # Update labels with initial values

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

# Create participant ID input window
participant_id_window = tk.Toplevel(root)
participant_id_window.title("Enter Participant ID")
tk.Label(participant_id_window, text="Participant ID:").pack(pady=10)
participant_id_entry = tk.Entry(participant_id_window)
participant_id_entry.pack(pady=10)
tk.Button(participant_id_window, text="Submit", command=get_participant_id).pack(pady=10)

# Run the main loop
root.mainloop()
