# IADAI-1000067-ASHUTOSH_SAMAL

Introduction to Pose Detection Project

For creating my pose detection project, I started with research on how pose detection works, exploring various techniques and libraries that could be applied to achieve my goal. After developing the basic structure of my Python code in Visual Studio Code (VS Code), I installed key extensions such as Jupyter Notebook and Python to ensure a seamless coding experience.

Next, I used the command prompt to install essential Python libraries. The ones I chose include numpy, pandas, opencv-python, tensorflow, and keras—each selected for a specific purpose in data processing, machine learning, and image processing. I also collected data for training and testing. For this, I downloaded twelve videos from YouTube, each representing a different gesture. I aimed to train the model on three key gestures: clapping, walking, and running. For each gesture, I selected four videos—three for training and one for testing.

After downloading, I converted each video to .avi format for easier processing. My next step was to create a dataset using these videos, normalizing the data, splitting it into training and test sets, and running accuracy tests on the model. My model ultimately achieved 100% accuracy on the training set. Finally, I ran my test code to validate the model, and everything worked as expected. The project can be terminated by pressing the ‘Q’ key to exit.

Code Explanation

This Python code captures frames from a video and extracts human pose landmarks using MediaPipe’s Pose module, which detects 33 specific points on the human body. These landmarks are saved into a CSV file, enabling the training of a machine learning model on the pose data.

Let's break down the code step-by-step.

Step 1: Import Libraries and Set Up

import cv2
import mediapipe as mp
import csv

In this section, we import the necessary libraries:

cv2 (OpenCV): This is a powerful library for computer vision tasks. Here, it’s used to read and process video frames.
mediapipe: A Google library for real-time perception tasks, which provides pre-trained models for pose estimation. The mp_pose module specifically helps in identifying 33 landmarks on the human body.

csv: This is the standard Python library for handling CSV files, which allows us to store and organize the data in a spreadsheet format.

Step 2: Initialize MediaPipe Pose Module

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

Here, we initialize mp_pose, which allows access to the Pose module in MediaPipe. The Pose() function creates an instance for processing images to detect human poses. Additionally, mp_drawing provides tools for visualizing detected landmarks directly on the images.

Step 3: Open Video File

cap = cv2.VideoCapture('eyeblink3.avi')

Using OpenCV's VideoCapture, the code opens the specified video file (eyeblink3.avi). This video file is part of the training data representing one of the gestures (in this case, blinking or possibly a similar facial or body action).

Step 4: Prepare the CSV File for Storing Data

with open('pose_landmarks.csv', mode='a', newline='') as f:
    csv_writer = csv.writer(f)

A CSV file called pose_landmarks.csv is created (or appended to if it already exists). This file will store each frame’s pose data, including landmark coordinates. csv_writer will write rows of data, with each row representing one frame.

Step 5: Define the CSV Header

headers = ['frame', 'label']
for i in range(33):
    headers += [f'x_{i}', f'y_{i}', f'z_{i}', f'visibility_{i}']
csv_writer.writerow(headers)

headers = ['frame', 'label']
for i in range(33):
    headers += [f'x_{i}', f'y_{i}', f'z_{i}', f'visibility_{i}']
csv_writer.writerow(headers)

The headers for the CSV file are created to describe each column of data. The first column is for the frame count, the second for labels, and the rest for the x, y, z coordinates and visibility status of each landmark. MediaPipe Pose has 33 landmarks, each with x, y, and z coordinates along with a visibility score. This loop creates these column names dynamically, such as x_0, y_0, z_0, visibility_0, etc., up to visibility_32.

Step 6: Process Each Frame

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

The while loop iterates over each frame in the video. The variable frame_count keeps track of the frame number, while cap.read() reads the frame from the video. If there are no more frames (or the video file ends), the loop breaks.

Step 7: Convert Frame to RGB and Detect Landmarks

image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)

MediaPipe’s Pose model works with RGB images. Therefore, cv2.cvtColor converts the frame from BGR (OpenCV’s default format) to RGB. The pose.process() function then analyzes the frame and returns detected landmarks, stored in results.

Step 8: Check for Detected Landmarks and Prepare CSV Data

if results.pose_landmarks:
    row = [frame_count, 'label_here']  # Use a placeholder label initially
    
    for landmark in results.pose_landmarks.landmark:
        row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    csv_writer.writerow(row)
If landmarks are detected in the frame, the code creates a new row with the current frame_count and a placeholder label ('label_here'). The code then loops over each landmark in results.pose_landmarks.landmark, extracting x, y, z, and visibility values, appending them to the row list. Once all landmark data is added, the row is written to the CSV file.

Step 9: Increment Frame Count

frame_count += 1

This increments the frame count to reflect that the program is moving to the next frame.

Step 10: Release Resources and Close MediaPipe

cap.release()
pose.close()

Once the loop completes (either the video ends or there are no more frames), cap.release() closes the video file, and pose.close() releases the resources used by the MediaPipe model. This step is essential for efficient memory management.

Summary of Project Workflow

Preparation and Research: Research was conducted to identify the best tools and methods for pose detection. The environment (VS Code) and necessary extensions were set up, and Python libraries were installed.

Data Collection: Twelve videos were downloaded, converted to .avi format, and used to create a dataset. Three gestures (clap, walk, run) were selected for model training and testing.

Data Processing and Model Training: The code captured landmarks from each frame in the videos, normalizing the data, splitting it into training and testing sets, and running accuracy tests. The model achieved 100% accuracy on the training set.

Testing and Results: The model was validated with test data, and everything ran smoothly.
