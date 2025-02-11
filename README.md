# 🎭 Face Detection & Recognition using OpenCV  

A real-time **Face Detection & Recognition** system using OpenCV and Deep Learning. This project detects faces in a webcam feed and recognizes them using a trained model.

## 🚀 Features  
✅ **Real-time Face Detection** using OpenCV's Deep Learning Model (DNN).  
✅ **Face Recognition** using **LBPH (Local Binary Patterns Histogram)**.  
✅ **Trains on New Faces** and saves them automatically.  
✅ **Works with Live Webcam** or Pre-stored Images.  
✅ **Stores Detected Faces** for future training.  

## 📸 Demo  
*(Add a screenshot or GIF of your project running here!)*  
<img src="demo.gif" width="500"/>  

## 🔧 Installation  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/yourusername/Face-Detection-Recognition.git
cd Face-Detection-Recognition
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the script
python face_detection_recognition.py
⚙️ How It Works
The script loads a pre-trained DNN model for face detection.
It captures video from the webcam in real-time.
It detects faces and saves them for training.
It recognizes known faces using the trained model.
Press 'q' to exit and train the model for future recognition.
📜 Requirements
Python 3.x
OpenCV
NumPy

