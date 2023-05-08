import face_recognition
import cv2
import numpy as np

# This project uses our webcam and tries to detect and, if possible, recognize the faces in view

# To speed things up, we will use 1/4 the resolution of our camera and skip every-other frame

video_capture = cv2.VideoCapture(0) # this references webcam #0 which is the default one

#load in our first sample face:
todd_image = face_recognition.load_image_file("todd.jpeg")
todd_face_encoding = face_recognition.face_encodings(todd_image)[0]

# Do the same for the second image:
damian_image = face_recognition.load_image_file("damian.jpg")
damian_face_encoding = face_recognition.face_encodings(damian_image)[0]

jiho_image = face_recognition.load_image_file("jiho.png")
jiho_face_encoding = face_recognition.face_encodings(jiho_image)[0]

aaron_image = face_recognition.load_image_file("aaron.png")
aaron_face_encoding = face_recognition.face_encodings(aaron_image)[0]

sarah_image = face_recognition.load_image_file("sarah.jpg")
sarah_face_encoding = face_recognition.face_encodings(sarah_image)[0]

ilan_image = face_recognition.load_image_file("ilan.jpg")
ilan_face_encoding = face_recognition.face_encodings(ilan_image)[0]

brian_image = face_recognition.load_image_file("brian.png")
brian_face_encoding = face_recognition.face_encodings(brian_image)[0]

# create an array of known face encodings and thei names
known_face_encodings = [todd_face_encoding, damian_face_encoding, jiho_face_encoding, aaron_face_encoding, sarah_face_encoding, ilan_face_encoding, brian_face_encoding]
known_face_names = ["Todd A.", "Damian M.", "Jiho S.", "Aaron A.", "Sarah K.", "Ilan V.", "Brian P."]

# setup some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True: # loop forever (until we quit)
    # grab a single frame from our webcam
    ret, frame = video_capture.read()

    if process_this_frame:
        # resize frame to 1/4 size for speed:
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        # convert the image from BGR (which OpenCV uses) to RGB which face_recognition uses
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1]) # reverses color order of every row and every column
        # find all the faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame) # look at shrunken image that we've turned into RGB and find all of the faces in it
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) 

        face_names = []
        for face in face_encodings:
            # see if this face matches our known faces:
            matches = face_recognition.compare_faces(known_face_encodings, face)
            name = "Unknown"
            # we will use the known_face with the smallest distance to the current face
            face_distances = face_recognition.face_distance(known_face_encodings, face)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index] # if this face matches the ones we have, put that name, if it doesn't match it will stay unknown
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # since we scaled our image down to 1/4, we need to multiply by 4 to get actual values
        top    *= 4
        right  *= 4
        bottom *= 4
        left   *= 4

        # Draw a red box around each face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow("Face Detection (type Q to quit)", frame)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the webcam
video_capture.realease()
cv2.destroyAllWindows()
