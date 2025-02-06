import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image

# Set up Streamlit app
st.title("ANPR on Image with Two YOLOv8 Models")

# File uploader
uploaded_file = st.file_uploader("Upload your image:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    # Convert image to RGB and save
    image_rgb = image.convert("RGB")
    image_rgb.save("saved_image.jpg", format="JPEG")
    frame = cv2.imread("saved_image.jpg")

    # Load the detection model (to locate text regions)
    detection_model = YOLO("runs/detect/train/weights/best.pt")  # Replace with your detection model path

    # Load the recognition model (to recognize text within the regions)
    recognition_model = YOLO("best.pt")  # Replace with your recognition model path

    # Run detection model to find text regions
    detection_results = detection_model(frame, save=True)

    # Check if any text regions are detected
    if not detection_results[0]:
        st.subheader("No text regions detected!")
    else:
        with st.spinner("Processing..."):
            for result in detection_results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, _ = result

                # Draw bounding box around the detected text region
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                # Crop the detected text region
                cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
                cv2.imwrite('cropped_text.jpg', cropped_img)
                cropped_image = Image.open('cropped_text.jpg')

                # Display the cropped text region
                st.image(cropped_image, caption='Detected Text Region')

              # Run the recognition model on the cropped text region using predict
                # Run the recognition model on the cropped text region using predict
                import streamlit as st

            # Assuming recognition_model is already loaded and cropped_img is the input image
            recognition_results = recognition_model.predict(cropped_img, conf=0.5, save=True)

            # Check if text is recognized
            if not recognition_results[0].boxes:
                st.write("No text recognized in this region.")
            else:
                class_labels = {
                    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                    10: 'A', 11: 'B', 12: 'waw', 13: 'D', 14: 'H', 15: 'w', 16: 'CH',
                }

                # Lists to separate numbers, letters, and others
                numbers = []
                letters = []
                others = []

                # Process detected characters
                for result in recognition_results:
                    for box in result.boxes:
                        cls = int(box.cls[0])  # Detected class
                        label = class_labels.get(cls, '')  # Get corresponding label
                        x_position = box.xyxy[0][0]  # Horizontal position for sorting

                        if cls <= 9:  # It's a number
                            numbers.append((x_position, label))
                        elif cls >= 10:  # It's a letter
                            letters.append((x_position, label))
                        else:  # Other characters (if any)
                            others.append((x_position, label))

                # Sort characters by their horizontal position
                numbers.sort(key=lambda x: x[0])
                letters.sort(key=lambda x: x[0])
                others.sort(key=lambda x: x[0])

                # Separate numbers into two parts: before and after letters
                if numbers and letters:
                    # Find the position of the first letter
                    first_letter_position = letters[0][0]

                    # Split numbers into two groups: before and after the first letter
                    numbers_before_letters = [num for num in numbers if num[0] < first_letter_position]
                    numbers_after_letters = [num for num in numbers if num[0] >= first_letter_position]
                else:
                    numbers_before_letters = numbers
                    numbers_after_letters = []

                # Extract text for numbers, letters, and others
                numbers_text = ''.join([char[1] for char in numbers_before_letters]) if numbers_before_letters else ''
                letters_text = ''.join([char[1] for char in letters]) if letters else ''
                others_text = ''.join([char[1] for char in numbers_after_letters]) if numbers_after_letters else ''

                # Build the plate in the desired format
                plate_text = f"{numbers_text} | {letters_text} | {others_text}"

                # Display the detected plate
                st.write(f"Detected Plate: {plate_text}")