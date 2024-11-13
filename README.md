# IADAI-1000067-ASHUTOSH_SAMAL

Introduction to Pose Detection Project

To create my pose detection project, I began by researching different methods for human pose estimation. After gaining a clear understanding, I developed my Python code in Visual Studio Code (VS Code). I set up the necessary extensions in VS Code, including Jupyter Notebook and Python, to streamline my workflow. Then, using the command prompt, I installed essential Python libraries like numpy, pandas, opencv-python, tensorflow, and keras, which were needed for data processing, image handling, and machine learning tasks. These tools and libraries provided a strong foundation for building and testing the pose detection model. Overall, this setup allowed me to approach pose detection efficiently and effectively using computer vision and AI techniques.

Next, I used the command prompt to install key Python libraries like numpy, pandas, opencv-python, tensorflow, and keras, each chosen for tasks like data processing, image handling, and machine learning. I gathered training and testing data by downloading twelve YouTube videos, focusing on three gestures: eyeblink, handshake, and smile. For each gesture, I used four videos—three for training and one for testing. This setup helped ensure accurate recognition of these specific gestures in the model.

After downloading the videos, I converted each one to .avi format for smoother processing. Then, I created a dataset from these videos, normalizing the data and splitting it into training and test sets. I ran accuracy tests on the model, achieving 100% accuracy on the training set. Finally, I tested the model using the test code, and it worked as expected. The project is designed to terminate by pressing the ‘Q’ key, providing a simple way to exit the program when testing is complete.

1. Capturing frames and extracting pose landmarks:
Involves using OpenCV to open video files and MediaPipe to detect pose landmarks. Each video frame is processed, and detected landmarks (like key body joints) are saved in a CSV file with the frame number and a label (e.g., eyeblink, handshake, smile). MediaPipe Pose detects 33 landmarks, including hands and eyes, and stores each landmark’s x, y, z coordinates along with a visibility score for further analysis.

2. Normalizing the data:
Data normalization involves adjusting the raw landmark data by using the left hip's position as a reference point. This ensures the model focuses on the body's relative movements rather than its absolute position in the frame. By normalizing in this way, the model can better interpret gestures and poses, regardless of the person’s location within the video. This step is essential for making the pose detection process more accurate and adaptable to different environments and positions.

3. Splitting the dataset:
To prepare the model, we split the normalized data into training and testing sets using the train_test_split() method. This method allows the model to learn patterns from one part of the data and then test its accuracy on the remaining portion. By setting aside 20% of the data for testing, we can evaluate the model’s performance objectively. This split is essential for ensuring the model generalizes well and accurately recognizes poses on new data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

4. Training the model:
I built a neural network in TensorFlow/Keras with two dense layers: the first with 64 neurons and the second with 32, both using the ReLU activation function. The output layer uses softmax to predict one of three gestures: clap, walk, or run. After compiling the model, I trained it on landmark data and achieved 100% accuracy on the test set. The model successfully learned to classify the gestures based on the given input data.

model = tf.keras.models.Sequential([ tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(len(set(y)), activation='softmax') ])

5. Saving the model:
After training, the model is saved to a file, allowing it to be reused later for testing or deployment without needing to retrain it each time.

6. Testing the model:
In the final step, we load the trained model to make predictions on new video data. For each frame, we detect the pose landmarks, process them for the model, and predict the gesture (clap, walk, or run). The predicted gesture is then displayed on the video screen in real-time, allowing for instant feedback during the test.

7. Terminating the program:
A new window opens to display the output, and the video can be stopped by pressing the Q key. To summarize, after setting up the environment and downloading the required videos, I created a dataset of pose landmarks for different gestures. I then trained a neural network model, tested it on new video data, and achieved 100% accuracy in recognizing gestures. The project is completed by pressing Q to close the video display.

Once the loop completes (either the video ends or there are no more frames), cap.release() closes the video file, and pose.close() releases the resources used by the MediaPipe model. This step is essential for efficient memory management.

Output:


For handshake:
![image](https://github.com/user-attachments/assets/649bd687-647e-4308-92ee-f9390c4c4d1c)

For Smile:
![image](https://github.com/user-attachments/assets/f5244f97-6a2e-43cd-b9bf-61c172539fde)


Link to github repositary: https://github.com/Insane350/IADAI-1000067-ASHUTOSH_SAMAL



here is the screenshot of the F1 score and confusion matrix:- 
![WhatsApp Image 2024-11-12 at 22 05 13_3f2f6874](https://github.com/user-attachments/assets/93880e2e-89f4-4cfa-8f51-7713b66fd56a)

