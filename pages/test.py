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
    detection_model = YOLO("path/to/your/detection_model.pt")  # Replace with your detection model path

    # Load the recognition model (to recognize text within the regions)
    recognition_model = YOLO("path/to/your/recognition_model.pt")  # Replace with your recognition model path

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

                # Run the recognition model on the cropped text region
                recognition_results = recognition_model(cropped_img, save=True)

                # Check if text is recognized
                if not recognition_results[0]:
                    st.write("No text recognized in this region.")
                else:
                    # Extract recognized text and confidence score
                    recognized_text = recognition_results[0].names  # Adjust based on your model's output format
                    confidence = recognition_results[0].boxes.conf  # Confidence score

                    # Display the recognized text and confidence score
                    st.write(f"Recognized text: {recognized_text}")
                    st.write(f"Confidence score: {confidence:.2f}")