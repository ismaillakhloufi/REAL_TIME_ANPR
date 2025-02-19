import cv2
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
from PIL import Image

st.title("ANPR on Video with Two YOLOv8 Models")

# File uploader for video
uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi"])

if "value_set" not in st.session_state:
    st.session_state.value_set = True

if st.session_state.value_set and uploaded_file is not None:
    st.video(uploaded_file)

    # Save uploaded video to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Load YOLO models
    detection_model = YOLO("Detection_model/best.pt")  # Model for plate detection
    recognition_model = YOLO("OCR_model/best.pt")  # Model for plate character recognition

    # Open video file
    cap = cv2.VideoCapture(temp_file.name)
    if not cap.isOpened():
        st.error("Error: Could not open video.")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_nbr = 0

    with st.spinner("Processing..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if the video ends

            # Process every 10th frame for efficiency
            if frame_nbr % 10 == 0:
                # Detect license plates
                detection_results = detection_model(frame)

                for result in detection_results[0].boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = result

                    if score > 0.5:
                        # Draw bounding box around the plate
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                        # Crop the detected plate region
                        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
                        cv2.imwrite("cropped_plate.jpg", cropped_img)

                        # Run recognition model on cropped plate
                        recognition_results = recognition_model.predict(cropped_img, conf=0.5)

                        if recognition_results and recognition_results[0].boxes:
                            # Define class labels (modify as needed)
                            class_labels = {
                                0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                                10: 'A', 11: 'B', 12: 'waw', 13: 'D', 14: 'H', 15: 'w', 16: 'CH',
                            }

                            # Lists to separate numbers, letters, and others
                            numbers = []
                            letters = []
                            others = []

                            for result in recognition_results:
                                for box in result.boxes:
                                    cls = int(box.cls[0])  # Detected class
                                    label = class_labels.get(cls, '')  # Get corresponding label
                                    x_position = box.xyxy[0][0]  # Horizontal position for sorting

                                    if cls <= 9:  # It's a number
                                        numbers.append((x_position, label))
                                    elif cls >= 10:  # It's a letter
                                        letters.append((x_position, label))
                                    else:  # Other characters
                                        others.append((x_position, label))

                            # Sort characters by horizontal position
                            numbers.sort(key=lambda x: x[0])
                            letters.sort(key=lambda x: x[0])
                            others.sort(key=lambda x: x[0])

                            # Separate numbers into two parts: before and after letters
                            if numbers and letters:
                                first_letter_position = letters[0][0]
                                numbers_before_letters = [num for num in numbers if num[0] < first_letter_position]
                                numbers_after_letters = [num for num in numbers if num[0] >= first_letter_position]
                            else:
                                numbers_before_letters = numbers
                                numbers_after_letters = []

                            # Extract text for numbers, letters, and others
                            numbers_text = ''.join([char[1] for char in numbers_before_letters]) if numbers_before_letters else ''
                            letters_text = ''.join([char[1] for char in letters]) if letters else ''
                            others_text = ''.join([char[1] for char in numbers_after_letters]) if numbers_after_letters else ''

                            # Format the detected license plate text
                            plate_text = f"{numbers_text} | {letters_text} | {others_text}"

                            # Display detected plate on Streamlit
                            st.write(f"Detected Plate: {plate_text}")

                            # Draw detected plate text on the frame
                            cv2.putText(frame, plate_text, (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Write the processed frame to output video
                out.write(frame)
                cv2.imshow("ANPR Processing", frame)

            frame_nbr += 1

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        st.session_state.value_set = False

if not st.session_state.value_set:
    # Provide download buttons for output files
    st.header("Download Processed Video and Data")

    # Video File Download
    video_file = open("output_video.mp4", "rb")
    video_bytes = video_file.read()
    video_file.close()

    st.download_button(
        label="Download Processed Video",
        data=video_bytes,
        file_name="output_video.mp4",
        mime="video/mp4",
    )
