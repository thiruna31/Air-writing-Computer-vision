import cv2
import numpy as np

def detect_shape(canvas):
    """
    Detect the shape drawn on the canvas.
    
    :param canvas: BGR image with drawings
    :return: Name of the detected shape: "Triangle", "Rectangle", "Square", "Circle", "Star", or "None"
    """
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Ignore small noise
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        sides = len(approx)

        # Classify shape based on number of sides
        if sides == 3:
            return "Triangle"
        elif sides == 4:
            # Optional: check aspect ratio to distinguish square vs rectangle
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.9 <= ar <= 1.1:
                return "Square"
            else:
                return "Rectangle"
        elif sides > 8:
            return "Circle"
        elif 5 <= sides <= 8:
            return "Star"

    return "None"
