import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

#loading the first criminal image who is me enock
criminal1_image = face_recognition.load_image_file("enock.jpg")
criminal1_face_encoding = face_recognition.face_encodings(criminal1_image)[0]

# Loading the second criminal image who is my team_project member
criminal2_image = face_recognition.load_image_file("faith.jpg")
criminal2_face_encoding = face_recognition.face_encodings(criminal2_image)[0]

# creating arrays of the two criminals
criminals_face_encodings = [
    criminal1_face_encoding,
    criminal2_face_encoding
]
criminals_names = [
    "ENOCK (wanted hacker)",
    "FEI"
]

# Initializing some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grabing a single frame of video
    ret, frame = video_capture.read()

    # Resizing frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Converting the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Finding all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(criminals_face_encodings, face_encoding)
            name = "citizen"

            #using the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(criminals_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = criminals_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Displaying the video results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scaling back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Drawing a box around the faces
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Drawing a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Displaying the resulting image
    cv2.imshow('KENYA POLICE CCTV CRIMINAL LOCATOR', frame)

    # if you Hit 't' on the keyboard it will quite
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Releasing handle to the webcam
video_capture.release()
cv2.destroyAllWindows()