import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- 1. 加载 MNIST 模型 ---
try:
    model = load_model('mnist_cnn_best_model.h5')
    print("MNIST model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. 预处理单个字符图像以适应 MNIST 模型 ---
def preprocess_char_image(char_img):
    """
    Preprocesses a single character image to be MNIST-compatible.
    - Assumes char_img is a grayscale image of a single character.
    """
    # Invert colors if necessary (MNIST often expects white digit on black background)
    # If your character is black on white after thresholding, you might need to invert.
    # Let's assume for now the segmentation gives us character in white, background in black
    # If not, invert: char_img = cv2.bitwise_not(char_img)

    # Resize to 28x28
    resized_char = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)

    # Ensure it's a single channel image (it should be already if input was grayscale)
    if len(resized_char.shape) == 3 and resized_char.shape[2] == 3: # if somehow it became color
        resized_char = cv2.cvtColor(resized_char, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to 0-1 (common for neural networks)
    normalized_char = resized_char.astype('float32') / 255.0

    # Reshape for model input: (1, 28, 28, 1) for a single sample
    # (batch_size, height, width, channels)
    reshaped_char = np.expand_dims(normalized_char, axis=-1) # Add channel dimension
    reshaped_char = np.expand_dims(reshaped_char, axis=0)    # Add batch dimension

    return reshaped_char

# --- 3. 主识别逻辑 ---
def recognize_captcha(image_path, model):
    # Load CAPTCHA image
    captcha_image_bgr = cv2.imread(image_path)
    if captcha_image_bgr is None:
        print(f"Error: Could not read image at {image_path}")
        return ""

    # --- a. 预处理验证码图像 ---
    # Convert to grayscale
    gray_image = cv2.cvtColor(captcha_image_bgr, cv2.COLOR_BGR2GRAY)

    # Thresholding to binarize the image (this is highly dependent on the CAPTCHA)
    # You might need to experiment with different thresholding methods and values
    # For simple CAPTCHAs, a global threshold might work.
    # For more complex ones, adaptive thresholding or Otsu's binarization is better.
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # THRESH_BINARY_INV assumes characters become white, background black.
    # If your characters are dark and background light, use THRESH_BINARY.

    # Optional: Noise removal (e.g., morphological operations)
    # kernel = np.ones((2,2),np.uint8)
    # thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel) # Remove small noise
    # thresh_image = cv2.dilate(thresh_image, kernel, iterations = 1) # Thicken characters

    cv2.imshow("Thresholded CAPTCHA", thresh_image) # Display for debugging
    cv2.waitKey(0)


    # --- b. 分割字符 (Contour-based segmentation - simple approach) ---
    contours, _ = cv2.findContours(thresh_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and sort contours (e.g., by x-coordinate for left-to-right reading)
    # This part is CRITICAL and often the hardest.
    # We need to filter out noise contours and get only character contours.
    bounding_boxes = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        # Add filters here based on expected character size, aspect ratio, area
        # Example: if w > 5 and h > 10 and w < 50 and h < 50: # Adjust these values
        # A common filter is contour area
        if cv2.contourArea(contour) > 50: # Tune this threshold
             bounding_boxes.append((x, y, w, h))

    if not bounding_boxes:
        print("No characters found after contour detection.")
        return ""

    # Sort bounding boxes by their x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    captcha_text = ""
    visualization_image = captcha_image_bgr.copy() # For drawing boxes

    # --- c. & d. 针对每个分割出的字符进行处理和预测 ---
    for (x, y, w, h) in bounding_boxes:
        # Extract the character from the thresholded image
        char_roi = thresh_image[y:y+h, x:x+w]

        # Draw rectangle on visualization image
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # Add padding to make it more square-like before resizing (optional, but can help)
        # This helps maintain aspect ratio when resizing to 28x28.
        # Determine padding
        pad_y = int((max(w,h) - h)/2)
        pad_x = int((max(w,h) - w)/2)
        char_padded = cv2.copyMakeBorder(char_roi, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0,0,0]) # Pad with black

        # Preprocess for MNIST model
        processed_char = preprocess_char_image(char_padded)

        # Make prediction
        prediction_probs = model.predict(processed_char)
        predicted_digit = np.argmax(prediction_probs)

        captcha_text += str(predicted_digit)
        print(f"Segment ({x},{y},{w},{h}): Predicted as {predicted_digit}, Probs: {prediction_probs.round(3)}")


    cv2.imshow("Segmented Characters", visualization_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return captcha_text

# --- 调用 ---
captcha_file_path = 'custom_images/test1.png'  # Make sure this path is correct
if captcha_file_path:
    recognized_text = recognize_captcha(captcha_file_path, model)
    print(f"\nRecognized CAPTCHA text for '{captcha_file_path}': {recognized_text}")
else:
    print("Please provide the path to the CAPTCHA image.")