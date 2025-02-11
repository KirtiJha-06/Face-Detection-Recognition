🔹 What is Face Detection?
Face Detection is the process of identifying human faces in an image or video. It does not recognize who the person is—it just detects that a face exists.
🔹 Example: When your phone camera detects faces before clicking a photo.

✅ In this project, we use a Deep Learning-based model (res10_300x300_ssd_iter_140000.caffemodel) to detect faces in real-time.

🔹 What is Face Recognition?
Face Recognition is the process of identifying who the detected face belongs to by comparing it with stored images.
🔹 Example: Face Unlock on smartphones.

✅ In this project, we use the LBPH (Local Binary Pattern Histogram) algorithm to recognize faces that were previously saved and trained.

🔍 How This Project Works (Step-by-Step)
1️⃣ Face Detection – Using a Deep Learning Model (DNN)
The project loads a pre-trained deep learning model called SSD (Single Shot Detector) to detect faces in real-time.
This model is trained on the ResNet architecture, making it more accurate than Haar Cascades.
The face is extracted and stored for training.

2️⃣ Face Recognition – Using Machine Learning (LBPH Algorithm)
After detecting a face, we convert it to grayscale and train it using the LBPH (Local Binary Patterns Histogram) algorithm.
LBPH is a simple but powerful machine learning model that:
Converts the face into a grid of pixels.
Detects patterns in those pixels.
Compares the pattern with stored face data.
If the model recognizes a face, it labels it (e.g., "Person 1"). Otherwise, it shows "Unknown".

🔬 Machine Learning Models Used
Task	                             Model Used                              	Why?
Face Detection :	  Deep Learning SSD (Single Shot Detector)  	More accurate & faster than Haar Cascades
Face Recognition: 	LBPH (Local Binary Pattern Histogram)	      Works well with small datasets, faster training

📌 Technologies Used
✅ Python – Main programming language
✅ OpenCV – For image processing
✅ NumPy – For numerical operations
✅ Deep Learning Model (DNN) – For face detection
✅ LBPH Algorithm – For face recognition

🎯 Real-World Applications
✔ Security Systems – Face-based authentication
✔ Attendance Tracking – Schools, offices, and events
✔ Smart Surveillance – Detect & recognize intruders
✔ Personal Devices – Face unlock in mobile phones
