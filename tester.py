import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Model
import cv2
import os
import time
TF_ENABLE_ONEDNN_OPTS=0


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

model=tf.keras.models.load_model(r"C:\Users\USER\Downloads\lstm_model_gpu_98.h5")

def extract_and_preprocess_frames(video_path, frame_rate=5):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract frames at specified frame_rate
    for frame_num in range(0, frame_count, frame_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            # Resize and preprocess the frame
            frame_resized = cv2.resize(frame, (224, 224))
            frame_features = extract_features(frame_resized)
            frames.append(frame_features)
        else:
            print(f"Warning: Could not read frame {frame_num} from {video_path}")


    cap.release()

    return np.array(frames)


def extract_features(frame):

    img = image.img_to_array(frame)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    features = cnn_model.predict(img)

    return features.flatten()

l=[3600,3700,3800]


for i in l:
    video_frames = extract_and_preprocess_frames(f"C://Users//USER//OneDrive//Documents//all_videos//IMG_{i}.mp4")
    pred_labels=['CUT','DRIVE','SWEEP']
    if video_frames is not None:

        num_frames = video_frames.shape[0]
        feature_dim = video_frames.shape[1]

        print(f"Number of frames: {num_frames}, Feature dimension: {feature_dim}")

        if feature_dim == 2048:
            video_frames_reshaped = video_frames.reshape(1, num_frames, feature_dim)

            # Make the prediction
            predictions = model.predict(video_frames_reshaped)
            print(f"Predictions for new video: {predictions}")

            print(f"this is a {pred_labels[np.argmax(predictions)]} shot")

        else:
            print(f"Feature dimension mismatch. Expected 2048 but got {feature_dim}.")

    video_path = f"C://Users//USER//OneDrive//Documents//all_videos//IMG_{i}.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_gap = 0.1 / cap.get(cv2.CAP_PROP_FPS)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # Read until video is completed
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Video PLayback', frame)
            time.sleep(frame_gap)
            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()