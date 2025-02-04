import cv2  # OpenCV for computer vision tasks
import mediapipe as mp  # MediaPipe for face mesh detection
import pyautogui  # For controlling the mouse cursor and clicks

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Initialize MediaPipe FaceMesh with refined landmarks for improved accuracy
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get the screen width and height for mapping cursor positions
screen_w, screen_h = pyautogui.size()

while True:
    # Capture a frame from the webcam
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror effect

    # Convert the frame to RGB format (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame with FaceMesh to get facial landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    # Get the dimensions of the frame
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        # Get landmarks for the first detected face
        landmarks = landmark_points[0].landmark

        # Loop through landmarks corresponding to the iris area (indices 474 to 477)
        for id, landmark in enumerate(landmarks[474:478]):
            # Convert normalized coordinates to pixel values
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            # Draw a small green circle at the iris landmarks
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

            if id == 1:  # Use the second iris point to control the mouse
                screen_x = screen_w * landmark.x  # Map x-coordinate to screen width
                screen_y = screen_h * landmark.y  # Map y-coordinate to screen height
                pyautogui.moveTo(screen_x, screen_y)  # Move the mouse cursor

        # Get landmarks for the left eye's top and bottom (indices 145 and 159)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)

            # Draw small yellow circles at the eye landmarks
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # Check for an eye blink (difference in y-coordinates is small)
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()  # Simulate a mouse click
            pyautogui.sleep(1)  # Add a delay to avoid repeated clicks

    # Display the video feed with overlays
    cv2.imshow('Eye Controlled Mouse', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
