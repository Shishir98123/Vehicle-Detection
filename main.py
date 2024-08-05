import cv2
import numpy as np
import time

# Video capture
cap = cv2.VideoCapture('video.mp4')

min_width_rectangle = 80
min_height_rectangle = 80
count_line_position = 550

# Initialize Background Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

detect = []
offset = 6  # Allowable error between pixels
counter = 0
line_color = (255, 127, 0)  # Initial line color
line_crossed_time = 0  # Timestamp of the last line cross

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    
    # Apply background subtraction
    img_sub = algo.apply(blur) 
    dilat = cv2.dilate(img_sub, np.ones((5, 5), np.uint8), iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if the line was crossed in the last 0.4 seconds
    if time.time() - line_crossed_time < 0.4:
        line_color = (0, 255, 0)  # Change line color to green if crossed
    else:
        line_color = (255, 127, 0)  # Reset line color to original

    cv2.line(frame, (25, count_line_position), (1200, count_line_position), line_color, 3)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_width_rectangle and h >= min_height_rectangle:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "VEHICLE: " + str(counter), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)

            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            for (cx, cy) in detect:
                if count_line_position - offset < cy < count_line_position + offset:
                    counter += 1
                    line_crossed_time = time.time()  # Set timestamp of line crossing
                    detect.remove((cx, cy))
                    print("Vehicle Counter: " + str(counter))
                    break  # Exit the loop after processing the crossing vehicle

    cv2.putText(frame, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    cv2.imshow('Video Original', frame)

    if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
        break

cv2.destroyAllWindows()
cap.release()
