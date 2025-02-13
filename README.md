# 🚗 Moroccan License Plate Detection and Recognition

## 📖 Overview

This project focuses on developing a computer vision system to detect and recognize Moroccan license plates using YOLO and Optical Character Recognition (OCR). By leveraging a dataset of Moroccan vehicle plates, the system aims to automate license plate recognition, overcoming challenges specific to Arabic characters and plate designs.
## Application Screenshot
![Application Screenshot](app_demo.png)

**👥 Team Members:**
- [LAKHLOUFI ISMAIL](https://github.com/ismaillakhloufi) :
- [BOURKI ACHRAF](https://github.com/BOURKI970/) : 
- [AMMI YOUSSEF](https://github.com/youssefammi123/) : 

## 🎯 Objectives
-  Implement object detection using **YOLO** for license plate segmentation.
-  Extract and recognize characters from Moroccan license plates with OCR.
-  Optimize performance for real-world scenarios with diverse lighting and backgrounds.

## 📂 Dataset
We utilized a publicly available dataset of Moroccan vehicle plates containing:
- 📷 **705 unique images**: Cars, trucks, motorcycles.
- 🔠 Arabic characters and digits for plate segmentation and OCR.
- 🎛️ Data augmentation techniques applied, expanding the dataset to **4,935 images**.

### 🚗 Moroccan License Plate Structure
![Moroccan License Plate Structure](Moroccan-license-plate-structure.jpg)

### 🔄 Data Preprocessing
-  **Segmentation**: Cropped images to isolate plates.
-  **Augmentation**: Applied noise, contrast adjustments, flipping, and rotations.
-  **Formats**: Data labeled in XML, CSV, and YOLO-specific text files.

## 🛠️ Methodology
1. **Plate Detection**:
   -  Used **YOLOv8** for efficient and accurate plate localization.
   -  Achieved **98.20% accuracy** on the test set.
2. **Character Recognition**:
  -  **YOLOv8** for simultaneous character detection and classification.
  -  Achieved **0.54% accuracy** on the test set.

## 💻 Tools and Technologies
- 💡 **Programming Languages**: Python
- ⚙️ **Frameworks**: YOLOv8, OpenCV
- 📚 **Libraries**: NumPy, Pandas, Matplotlib
- 🔧 **Others**: Data augmentation with Python scripts

## Présentation
This is our [presentation PDF](https://www.canva.com/design/DAGeLzc6G8w/xMNAD96WYagAGu5DdXI1eA/edit?utm_content=DAGeLzc6G8w&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) for more details.


## 🖥️ How to start
 **Clone the Repository** :
   ```bash
  git clone https://github.com/your-repo/moroccan-license-plate-detection.git
cd moroccan-license-plate-detection
   ```

## 📜 Licence
This project is licensed under the [MIT License](LICENSE).  

