import cv2
import numpy as np


# Function to count fingers based on convexity defects
def count_fingers(contour):
    # Get the convex hull of the hand contour
    hull = cv2.convexHull(contour)

    # Calculate convexity defects
    convexity_defects = cv2.convexityDefects(hull, contour)
    fingers_up = 0

    if convexity_defects is not None:
        for defect in convexity_defects:
            s, e, f, d = defect[0]
            # If the depth of the defect is greater than a threshold, count it as a finger
            if d > 10000:
                fingers_up += 1

    return fingers_up


# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to get a binary image
    _, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found
    if contours:
        # Get the largest contour (assuming it's the hand)
        hand_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(hand_contour, True)
        hand_contour = cv2.approxPolyDP(hand_contour, epsilon, True)

        # Count fingers
        fingers_count = count_fingers(hand_contour)

        # Draw the contour and convex hull
        hull = cv2.convexHull(hand_contour)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)

        # Display the number of fingers detected
        cv2.putText(frame, f"Fingers: {fingers_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the output frame
    cv2.imshow("Hand Gesture Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
