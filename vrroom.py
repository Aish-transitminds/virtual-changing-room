import cv2
import imutils
import numpy as np
import os
import random


def loadImages():
    folder = "tshirt" 
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def virtual():
    # Ensure the output directory exists for snapshots
    if not os.path.exists('output'):
        os.makedirs('output')

    cap = cv2.VideoCapture(0)  # Initialize webcam
    images = loadImages()  # Load clothing images

    if not images:
        print("Error: No images loaded. Ensure 'tshirt' folder exists with images.")
        return

    thresholds = [130, 40, 75, 130]  # Thresholds for mask generation
    current_cloth_id = 0  # Start with the first clothing item
    threshold = thresholds[current_cloth_id]
    size = 180  # Initial size of the clothing overlay

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access the camera.")
            break

        frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect
        clothing_item = images[current_cloth_id]

        # Resize the frame
        resized_frame = imutils.resize(frame, width=800)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Detect circles (e.g., for head detection)
        circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1.2, 100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if r > 30:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    size = r * 7

        size = max(100, min(350, size))  # Constrain clothing size
        clothing_item = imutils.resize(clothing_item, width=size)

        frame_height, frame_width = frame.shape[:2]
        clothing_height, clothing_width = clothing_item.shape[:2]
        height = int(frame_height / 2 - clothing_height / 2)
        width = int(frame_width / 2 - clothing_width / 2)

        clothing_gray = cv2.cvtColor(clothing_item, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(clothing_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)

        roi = frame[height:height + clothing_height, width:width + clothing_width]
        background = cv2.bitwise_and(roi, roi, mask=mask_inv)
        foreground = cv2.bitwise_and(clothing_item, clothing_item, mask=mask)

        frame[height:height + clothing_height, width:width + clothing_width] = cv2.add(background, foreground)

        # Display instructions on the screen
        cv2.putText(frame, "Press 'n' for next, 'p' for previous, 'c' for snapshot, 'Esc' to exit", 
                    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)

        cv2.imshow('Virtual Dressing Room', frame)

        # Handle key presses
        key = cv2.waitKey(10) & 0xFF
        if key == ord('n'):  # Next clothing item
            current_cloth_id = (current_cloth_id + 1) % len(images)
            threshold = thresholds[current_cloth_id]
        elif key == ord('p'):  # Previous clothing item
            current_cloth_id = (current_cloth_id - 1) % len(images)
            threshold = thresholds[current_cloth_id]
        elif key == ord('c'):  # Capture snapshot
            snapshot_name = f'output/snapshot_{random.randint(1, 999999)}.png'
            cv2.imwrite(snapshot_name, frame)
            print(f"Snapshot saved: {snapshot_name}")
        elif key == 27:  # Exit on 'Esc' key
            print("Exiting Virtual Dressing Room.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the virtual dressing room
virtual()
