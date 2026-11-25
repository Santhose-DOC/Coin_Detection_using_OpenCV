# Workshop 4 -- Coin Detection Using OpenCV in Python

## Aim

To detect and count coins in an input image using OpenCV by applying
grayscale conversion, Gaussian blurring, Canny edge detection, contour
detection, and minimum enclosing circles.

## Procedure

1.  Import the required libraries: OpenCV (`cv2`), NumPy, Matplotlib,
    and `os`.
2.  Specify the input image path as `coin.jpg`.
3.  Load the image using `cv2.imread()`. If the image cannot be loaded,
    raise a `FileNotFoundError`.
4.  Convert the original image from BGR to grayscale using
    `cv2.cvtColor()`.
5.  Apply Gaussian blur with an 11 × 11 kernel to reduce noise.
6.  Detect edges in the blurred image using the Canny edge detector with
    threshold values 30 and 150.
7.  Find the external contours from the detected edges using
    `cv2.findContours()`.
8.  Create a copy of the original image for displaying the detection
    results.
9.  Sort the contours by area and ignore contours with an area below
    300.
10. Calculate the minimum enclosing circle for each remaining contour.
11. Consider a contour as a coin when its enclosing-circle radius is
    greater than 15 pixels.
12. Draw a green circle around each detected coin and place a red number
    at its centre.
13. Print the total number of detected coins.
14. Display the original image and the detected-coin image side by side.
15. Display the processing stages: original image, grayscale image,
    blurred image, Canny edge detection, and final coin detection.

## Program

``` python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Specify the input image path
image_path = "coin.jpg"

# Load image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur and detect edges
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
edges = cv2.Canny(blurred, 30, 150)

# Find external contours
contours, _ = cv2.findContours(
    edges.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Filter contours, draw bounding circles, and count coins
output = image.copy()
coin_count = 0

# Sort contours by area
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for contour in contours:
    # Filter out small noise contours
    area = cv2.contourArea(contour)
    if area < 300:
        continue

    # Calculate minimum enclosing circle
    ((x, y), radius) = cv2.minEnclosingCircle(contour)

    # Filter by radius to identify coin-sized objects
    if radius > 15:
        coin_count += 1

        # Draw a green circle around the coin
        cv2.circle(
            output,
            (int(x), int(y)),
            int(radius),
            (0, 255, 0),
            3
        )

        # Draw a red number at the centre
        cv2.putText(
            output,
            f"{coin_count}",
            (int(x - 10), int(y + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

print(f"Total coins detected: {coin_count}")

# Display original image and detected coins
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Detected Coins: {coin_count}")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()

# Display all image-processing stages
fig, axs = plt.subplots(1, 5, figsize=(20, 6))

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(gray, cmap="gray")
axs[1].set_title("Grayscale Image")
axs[1].axis("off")

axs[2].imshow(blurred, cmap="gray")
axs[2].set_title("Blurred Image")
axs[2].axis("off")

axs[3].imshow(edges, cmap="gray")
axs[3].set_title("Edge Detection (Canny)")
axs[3].axis("off")

axs[4].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
axs[4].set_title(f"Detected Coins: {coin_count}")
axs[4].axis("off")

plt.tight_layout()
plt.show()
```

## Output

1.  **Original Image:** Displays the input coin image loaded from
    `coin.jpg`.
2.  **Grayscale Image:** Displays the input image converted into
    grayscale.
3.  **Blurred Image:** Displays the grayscale image after applying
    Gaussian blur to reduce noise.
4.  **Edge Detection (Canny):** Displays the edges identified using the
    Canny edge detector.
5.  **Detected Coins:** Displays the original image with green circles
    around detected coins and red numerical labels at their centres.
6.  **Total Coin Count:** Prints the total number of coins detected by
    the program.

*Note: The detected coin count depends on the input image and the
selected contour-area and radius thresholds. Keep `coin.jpg` in the
notebook's working directory before running the program.*

## Result

Thus, coins were detected and counted successfully using OpenCV. The
image-processing stages, including grayscale conversion, Gaussian
blurring, Canny edge detection, and contour analysis, were performed,
and the detected coins were marked with circles and numerical labels.
