
# import cv2
# import dlib
# import numpy as np
# from imutils import face_utils
# from scipy.spatial import distance as dist



# #setting these paths in a way that'll be easier for me to keep track of and type as i need to use them
# predictor_path = r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\FinalYear\ComputingProject\finalYearProject\BioPassTestCode\models\shape_predictor_68_face_landmarks.dat" #using this pre trained dlib library to make things a little easier
# haarcascade_path = r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\FinalYear\ComputingProject\finalYearProject\BioPassSource\data\haarcascade_frontalface_default.xml" #using haarcascade

# # Initialize face detection and landmark predictor
# face_cascade = cv2.CascadeClassifier(haarcascade_path)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)

# # Indices for eyes in the 68-point landmark model
# LEFT_EYE = list(range(36, 42))
# RIGHT_EYE = list(range(42, 48))

# # Function to calculate the Eye Aspect Ratio (EAR)
# def calculate_ear(eye):
#     # Vertical distances
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     # Horizontal distance
#     C = dist.euclidean(eye[0], eye[3])
#     # EAR formula
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Blink detection thresholds
# EYE_AR_THRESH = 0.25
# EYE_AR_CONSEC_FRAMES = 3
# blink_counter = 0
# total_blinks = 0

# # using the internal camera on my laptop and initialising it
# video = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video.read()
#     if not ret:
#         print("Frame capture unsuccessful")
#         break

#     # Convert frame to grayscale for detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces using Haar Cascade
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     # Draw rectangles around detected faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (252, 237, 172), 2)

#     # Detect faces using dlib for landmark prediction
#     dlib_faces = detector(gray)
#     for face in dlib_faces:
#         # Get landmarks
#         landmarks = predictor(gray, face)
#         landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

#         # Extract eye regions
#         left_eye = [landmarks[i] for i in LEFT_EYE]
#         right_eye = [landmarks[i] for i in RIGHT_EYE]

#         # Calculate EAR for both eyes
#         left_ear = calculate_ear(left_eye)
#         right_ear = calculate_ear(right_eye)
#         avg_ear = (left_ear + right_ear) / 2.0

#         # Check if EAR is below the threshold (blink detected)
#         if avg_ear < EYE_AR_THRESH:
#             blink_counter += 1
#         else:
#             if blink_counter >= EYE_AR_CONSEC_FRAMES:
#                 total_blinks += 1
#                 blink_counter = 0

#         # Display total blink count on the frame
#         cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Show the video frame with detections and blink counts
#     cv2.imshow("Liveness Detection", frame)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all OpenCV windows
# video.release()
# cv2.destroyAllWindows()


#import tensorflow as tf
import cv2
import dlib
import numpy as np
import face_recognition

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from scipy.spatial import distance as dist


# Paths for required models
predictor_path = r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\FinalYear\ComputingProject\finalYearProject\BioPassTestCode\models\shape_predictor_68_face_landmarks.dat"
haarcascade_path = r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\FinalYear\ComputingProject\finalYearProject\BioPassSource\data\haarcascade_frontalface_default.xml"


# Load the model without compiling
emotion_model = load_model(
    r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\Desktop\finalYearProject\BioPassSource\models\emotion_model.hdf5", 
    compile=False  # Load the model without the optimizer
)

# Load pre-trained emotion recognition model (using emotion_model.hdf5)
#emotion_model = load_model(r"C:\Users\AishaNtuli\OneDrive - National College of Ireland\Desktop\finalYearProject\BioPassSource\models\emotion_model.hdf5")
# Replace the optimizer (manually) with the updated one because the use of 'lr' is outdated and the correct argument is now 'learning rate'

emotion_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize face detection and facial landmark predictor
face_cascade = cv2.CascadeClassifier(haarcascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Indices for eyes in the 68-point landmark model
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Blink detection thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
blink_counter = 0
total_blinks = 0

# Emotion list for classification (make sure these match the new model's output categories)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to preprocess face for emotion detection
# def preprocess_face(img, face_location):
#     top, right, bottom, left = face_location
#     face = img[top:bottom, left:right]
#     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB as required by the model
#     face = cv2.resize(face, (48, 48))  # Resize to the model's expected input size
#     face = np.expand_dims(face, axis=0)  # Add batch dimension
#     face = face / 255.0  # Normalize pixel values to [0, 1]
#     return face

# Function to preprocess face for emotion detection
def preprocess_face(img, face_location):
    top, right, bottom, left = face_location
    face = img[top:bottom, left:right]
    
    # Convert the face to grayscale
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Resize the face to 64x64 pixels
    face = cv2.resize(face, (64, 64))  # Resize to 64x64
    
    # Normalize the face (model expects values between 0 and 1)
    face = np.expand_dims(face, axis=-1)  # Add channel dimension (64x64x1)
    face = np.expand_dims(face, axis=0)  # Add batch dimension (1, 64, 64, 1)
    face = face / 255.0  # Normalize the pixel values

    return face


# Combined function for blink and emotion-based authentication
def authenticate_user():
    global blink_counter, total_blinks
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale and RGB
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Blink Detection
        dlib_faces = detector(gray)
        for face in dlib_faces:
            landmarks = predictor(gray, face)
            landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            left_eye = [landmarks[i] for i in LEFT_EYE]
            right_eye = [landmarks[i] for i in RIGHT_EYE]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1
                    blink_counter = 0

            # Display blink count
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Emotion Detection and Authentication
        face_locations = face_recognition.face_locations(rgb_frame)
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Process face for emotion recognition
            face = preprocess_face(frame, face_location)
            prediction = emotion_model.predict(face)
            max_index = np.argmax(prediction[0])
            emotion = emotions[max_index]

            # Display detected emotion
            cv2.putText(frame, emotion, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Authentication logic
            if emotion == 'Happy' and total_blinks >= 3:  # Example logic: smile and blink at least 3 times
                cv2.putText(frame, "Authenticated!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Authentication Successful!")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Display the resulting frame
        cv2.imshow('Liveness and Emotion Authentication', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the authentication process
authenticate_user()

