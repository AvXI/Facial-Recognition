import cv2
import dlib

# Load the face detection model
detector = dlib.get_frontal_face_detector()

# Load the face recognition model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load the known face images and names
known_face_images = ['person1.jpg', 'person2.jpg', 'person3.jpg']
known_face_names = ['Person 1', 'Person 2', 'Person 3']
known_face_descriptors = []

# Calculate the face descriptors for the known faces
for image_path in known_face_images:
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    face_descriptor = face_recognizer.compute_face_descriptor(gray_image, faces[0])
    known_face_descriptors.append(face_descriptor)

# Open the video capture device
capture_device = cv2.VideoCapture(0)

# Initialize the video writer for saving the output
output_size = (int(capture_device.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(capture_device.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, output_size)

# Process each frame of the video capture device
while True:
    # Capture a frame from the video capture device
    ret, frame = capture_device.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = detector(gray_frame)

    # Process each detected face
    for face in faces:
        # Calculate the face descriptor for the current face
        face_descriptor = face_recognizer.compute_face_descriptor(gray_frame, face)

        # Find the closest matching known face descriptor
        distances = []
        for known_face_descriptor in known_face_descriptors:
            distance = dlib.distance(face_descriptor, known_face_descriptor)
            distances.append(distance)
        min_distance = min(distances)
        min_distance_index = distances.index(min_distance)

        # Draw a rectangle around the face and display the name of the person
        name = known_face_names[min_distance_index]
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write the frame to the video writer
    output_writer.write(frame)

    # Display the frame
    cv2.imshow('Advanced Facial Recognition', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and video writer
capture_device.release()
output_writer.release()

# Close all windows
cv2.destroyAllWindows()