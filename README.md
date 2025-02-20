

# 🚗 Moroccan License Plate Detection and Recognition

## 📖 Overview
This project focuses on developing a computer vision system to detect and recognize Moroccan license plates using YOLO and Optical Character Recognition (OCR). By leveraging a dataset of Moroccan vehicle plates, the system aims to automate license plate recognition, overcoming challenges specific to Arabic characters and plate designs.

![Application Screenshot](app_demo.png)

---

## 👥 Team Members
- **[LAKHLOUFI ISMAIL](https://github.com/ismaillakhloufi)**  
- **[BOURKI ACHRAF](https://github.com/BOURKI970/)**  
- **[AMMI YOUSSEF](https://github.com/youssefammi123/)**  

---

## 🎯 Objectives
- Implement object detection using **YOLOv8** for license plate segmentation.
- Extract and recognize characters from Moroccan license plates with OCR.
- Optimize performance for real-world scenarios with diverse lighting and backgrounds.

---

## 📂 Dataset
We utilized two publicly available datasets of Moroccan vehicle plates:

1. **[Moroccan License Plates Dataset for Detection](https://www.kaggle.com/datasets/ismaillakhloufi/moroccan-license-plates-dataset)**  
   - Contains **705 unique images** (cars, trucks, motorcycles).
   - Annotated for license plate detection in YOLO format.

2. **[Moroccan License Plates Characters for OCR](https://cc.um6p.ma/cc_datasets)**  
   - Includes **2,500+ images** (Arabic letters included).
   - Preprocessed for OCR training.

### 🔄 Data Preprocessing
- **Segmentation**: Cropped images to isolate plates.
- **Augmentation**: Applied noise, contrast adjustments, flipping, and rotations.
- **Formats**: Data labeled in XML, CSV, and YOLO-specific text files.

---

## 🛠️ Methodology
1. **Plate Detection**:
   - Trained **YOLOv8** for plate localization.
   - Achieved **98.20% accuracy** on the test set.
2. **Character Recognition**:
   - Fine-tuned **YOLOv8** for simultaneous character detection and classification.
   - Achieved **96.54% accuracy** on the test set.

---

## 💻 Tools and Technologies
- **Programming Languages**: Python  
- **Frameworks**: YOLOv8, OpenCV  
- **Libraries**: NumPy, Pandas, Matplotlib, Streamlit  
- **Utilities**: Data augmentation scripts, LabelImg for annotation  

---

## 📄 Presentation
[![Canva Presentation](https://img.shields.io/badge/Canva-Presentation-blue)](https://www.canva.com/design/DAGeLzc6G8w/xMNAD96WYagAGu5DdXI1eA/edit)

---

## 🖥️ How to Use

### Prerequisites
- Python 3.8+
- Git

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ismaillakhloufi/REAL_TIME_ANPR.git
   cd REAL_TIME_ANPR
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

3. **Change Directory to the app**:
   ```bash
   cd ANPR_Yolov8
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit App**:
   ```bash
   streamlit run 1_👋_main.py
   ```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
```

