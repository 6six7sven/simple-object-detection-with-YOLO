# Real-Time Object Detection with YOLO (You Only Look Once)

This Python script leverages OpenCV to perform real-time object detection using a pre-trained YOLO model.

## Functionality

- Utilizes OpenCV (`cv2`) for computer vision tasks.
- Employs NumPy (`numpy`) for numerical operations.
- Leverages a pre-trained YOLO model for object detection.
- Identifies objects from classes defined in "coco.names".
- Captures video from the webcam for continuous processing.
- Displays bounding boxes and labels for detected objects.

### Approach

1. **Import Libraries:** Imports `cv2` and `numpy`.
2. **Load YOLO Model:** Loads pre-trained model weights and configuration files using `cv2.dnn.readNet`.
3. **Define Object Classes:** Loads object class names from "coco.names" into a list.
4. **Identify Output Layers:** Extracts the names of network layers and pinpoints the output layers responsible for predictions.
5. **Capture Video:** Initializes webcam capture using `cv2.VideoCapture(0)`.
6. **Main Loop:**
    - **Read Frame:** Captures a frame from the webcam.
    - **Convert to Blob:** Prepares the frame for neural network input using `cv2.dnn.blobFromImage`.
    - **Perform Detection:** Passes the blob through the network with `net.forward` to obtain detections.
    - **Process Detections:**
        - Iterates through detections.
        - Selects class with highest confidence score.
        - Filters detections based on a confidence threshold (default: 0.5).
        - **If Object Detected:**
            - Calculates bounding box coordinates.
            - Draws bounding box on the frame using `cv2.rectangle`.
            - Adds label with detected class name using `cv2.putText`.
    - **Display Frame:** Shows the processed frame with OpenCV's `cv2.imshow`.
7. **Exit:** Terminates the script on 'q' key press, releasing the webcam and closing OpenCV windows.

### Usage

1. **Install Dependencies:**

   ```bash
   pip install opencv-python numpy
   ```

2. **Download Pre-Trained YOLO Model:**
   - Acquire the weights and configuration files for your desired YOLO model (e.g., YOLOv3, YOLOv5).
   - Place them in the same directory as your script.
3. **Run the Script:**

   ```bash
   python your_script_name.py
   ```

### Extensibility

- Implement sophisticated detection handling (object tracking across frames).
- Experiment with different object detection models.

### Disclaimer

This script serves as a fundamental example of real-time object detection. The effectiveness and accuracy depend on the chosen YOLO model and dataset.
