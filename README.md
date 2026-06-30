# CodeAlpha AI/ML Internship - 4 Tasks

This repository contains all 4 tasks completed for the CodeAlpha AI/ML Internship. Each task demonstrates a practical application of Artificial Intelligence and Machine Learning.

## ⚙️ Setup & Installation

**Important:** Task 1, 2, and 3 were developed and run on `Google Colab`. Task 4 was developed on `VS Code` with Python 3.11. All 4 tasks are provided in `.py` format for direct execution in VS Code or any Python IDE.

Install all required libraries for all 4 tasks with one command:
```bash
pip install ultralytics deep-sort-realtime opencv-python scikit-learn nltk music21 tensorflow "googletrans==4.0.0-rc1" "httpx==0.13.3"
---

### *1. Task 1: Language Translation Tool* `CodeAlpha_Task1_Language_Translation_Tool.py`

Python application that translates text between 100+ languages using Google Translate API.

#### *Features:*
- Real-time text translation
- Support for 100+ languages 
- Terminal-based input. Original Colab version had `ipywidgets GUI`.

#### *⚠️ Note*
This is the `.py` version for VS Code/Terminal. The original GUI version was in `Google Colab .ipynb`. Translation logic is 100% same.

#### *How to Run:*
1.  *Install Requirements:*
    pip install "googletrans==4.0.0-rc1" "httpx==0.13.3"
2.  *Run the file:*
    python CodeAlpha_Task1_Language_Translation_Tool.py
3.  *Input:* Enter text and language codes in terminal. e.g. `en` to `ur`.

#### *Demo Video:*
See `CodeAlpha_Task1_Language_Translation_ScreenRecording.zip` for working demonstration.

---

### *2. Task 2: FAQ Chatbot for E-commerce* `CodeAlpha_Task2_Chatbot_for_FAQs.py`

An AI-powered chatbot using TF-IDF and Cosine Similarity to answer customer questions.

#### *Features:*
- Answers FAQs in English and Urdu
- Uses `Scikit-learn` for text matching
- Runs directly in terminal

#### *How to Run:* 
1.  *Install Requirements:*
    pip install scikit-learn nltk
2.  *Run the file:*
    python CodeAlpha_Task2_Chatbot_for_FAQs.py
3.  *Input:* Type your question in the terminal. Type `exit` to quit.

#### *Demo Video:*
See `CodeAlpha_Task2_Chatbot_for_FAQs_ScreenRecording.zip` for working demonstration.

---

### *3. Task 3: Music Generation with AI* `CodeAlpha_Task3_Music_Generation_with_AI.py`

An LSTM neural network trained on MIDI data to generate new music sequences.

#### *Features:*
- Generates new `.mid` music files
- Uses `TensorFlow/Keras` and `music21` library
- No GUI needed

#### *How to Run:* 
1.  *Install Requirements:*
    pip install tensorflow music21 numpy
2.  *Run the file:*
    python CodeAlpha_Task3_Music_Generation_with_AI.py
3.  *Output:* A new `generated_music.mid` file will be saved in the folder.

#### *Demo Video:*
See `CodeAlpha_Task3_Music_Generation.zip` for demonstration.

#### **Files Included:**
- `CodeAlpha_Task3_Music_Generation_with_AI.py` : Main code file
- `CodeAlpha_Task3_Music_Generation.zip` : Screen recording 
- `CodeAlpha_TASK3_Generated_Music.mp3.mid` : AI generated music file. Open it in any media player.
---

### *4. Task 4: Object Detection & Tracking* `CodeAlpha_Task4_Object_Detection_Tracking.py`

Detects and tracks multiple objects in a video using YOLOv8 and DeepSORT.

#### *Features:*
- Real-time object detection with unique IDs
- Colored bounding boxes for each object - Category Based
- Saves output as a new video file

#### *A. Video File Version - How to Run:* 
1.  *Install Requirements:*
    pip install ultralytics deep-sort-realtime opencv-python
2.  *Requirement:* Place your `CodeAlpha_Task4_Video.zip` file in the same folder.
3.  *Run the file:*
    python CodeAlpha_Task4_Object_Detection_Tracking.py
4.  *Output:* `CodeAlpha_Task4_Output_Tracking.zip` will be created. Press `q` to exit.

#### *B. Webcam Version - How to Run:* 
1.  *File:* `CodeAlpha_Task4_Webcam_Tracking.py` 
2.  *Run the file:*
    python CodeAlpha_Task4_Webcam_Tracking.py
3.  *Output:* `CodeAlpha_Task4_Webcam_Recording.zip` will be created. Press `q` to exit.

#### *Demo Video:*
1.  See `CodeAlpha_Task4_Object_Tracking_ScreenRecording.zip` - Video File Demo
2.  See `CodeAlpha_Task4_Webcam_ScreenRecording.zip` - Webcam Live Demo

#### **Files Included:**
1- `CodeAlpha_Task4_Object_Detection_Tracking.py` : Main code file - Video Input
2- `CodeAlpha_Task4_Object_Detection_Tracking_Webcam.py` : Webcam code file - Live Input
3- `CodeAlpha_Task4_Video.zip` : Input video file used for detection
4- `CodeAlpha_Task4_Output_Tracking.zip` : Output video with detected and tracked objects
5- `CodeAlpha_Task4_Webcam_Recording.zip` : Output video recorded from webcam
6- `CodeAlpha_Task4_Object_Tracking_ScreenRecording.zip` : Screen recording - Video File
7- `CodeAlpha_Task4_Object_Tracking_Webcam_ScreenRecording.zip` : Screen recording - Webcam Live

###*About:*
This project is part of CodeAlpha Internship Python Programming Track


