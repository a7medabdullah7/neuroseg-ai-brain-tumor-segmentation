# NeuroSeg AI
## Brain Tumor MRI Segmentation Platform

**AI-Powered Medical Image Segmentation Using Deep Learning**

NeuroSeg AI is a deep learning-based web application designed to segment suspected brain tumor regions from MRI images using a pretrained U-Net architecture with a ResNet34 encoder.

The platform provides an interactive interface where users can upload MRI scans, run model inference, visualize predicted segmentation masks, and examine the resulting tumor regions.

> **Important:** NeuroSeg AI is an academic research and demonstration project. It is not a certified medical device and must not be used as a substitute for professional medical diagnosis.

---

# 1. Project Overview

The project focuses on automated brain tumor segmentation using Magnetic Resonance Imaging (MRI).

The main objective is to identify and delineate tumor regions at the pixel level using deep learning.

### Main Features

- Upload brain MRI images.
- Preview uploaded MRI scans.
- Perform automatic tumor segmentation.
- Visualize predicted segmentation masks.
- Display MRI and segmentation overlays.
- View image information and model configuration.
- Download segmentation results.

### Model Architecture

| Parameter | Configuration |
|---|---|
| Model | U-Net |
| Encoder | ResNet34 |
| Encoder Weights | ImageNet Pretrained |
| Input Channels | 1 (Grayscale) |
| Output Classes | 1 |
| Input Image Size | 256 × 256 |
| Activation | None (Logits) |
| Prediction Threshold | 0.60 |

---

# 2. Model Evaluation Results

The selected pretrained U-Net model achieved the following results on the held-out test dataset:

| Evaluation Metric | Result |
|---|---:|
| Positive Dice Score | 72.71% |
| Positive IoU | 62.13% |
| Empty Mask False Positive Rate | 26.57% |
| Best Validation Epoch | 4 |

**Metric Definitions:**

- **Positive Dice Score:** Measures the overlap between predicted and ground-truth tumor regions on images containing tumors.
- **Positive IoU:** Measures the intersection over union between predicted and ground-truth tumor regions on positive images.
- **Empty Mask FPR:** Measures the percentage of empty-mask test images for which the model predicts at least one positive pixel.

These metrics describe the model's performance on the project test set and do not guarantee equivalent performance on external clinical data.

---

# 3. Technologies and Libraries

The project was developed using the following technologies:

### Programming Language
- Python 3.10

### Deep Learning
- PyTorch
- Torchvision
- Segmentation Models PyTorch

### Web Application
- Streamlit

### Image Processing
- Pillow (PIL)
- NumPy

### Data Analysis and Visualization
- Pandas
- Matplotlib

### Model Architecture
- U-Net
- ResNet34 Encoder
- ImageNet Transfer Learning

### Development Environment
- Visual Studio Code
- Jupyter Notebook
- CUDA-enabled NVIDIA GPU

---

# 4. System Requirements

### Software Requirements

| Requirement | Recommended Configuration |
|---|---|
| Operating System | Windows 10 / Windows 11 |
| Python | 3.10.x |
| Package Manager | pip |
| Code Editor | Visual Studio Code (optional) |
| Web Browser | Google Chrome or Microsoft Edge |
| CUDA | Compatible NVIDIA CUDA environment (optional) |

### Hardware Requirements

**Recommended:**
- NVIDIA GPU with CUDA support.
- 8 GB RAM or more.
- Sufficient disk space for model weights and dependencies.

**CPU Mode:**

The application can also run without an NVIDIA GPU if the application code supports CPU inference. Processing may be slower.

---

# 5. Project Structure

```text
brain-tumor-segmentation/
│
├── .venv/
│
├── assets/
│
├── models/
│   ├── brain_tumor_pretrained_unet_resnet34.pth
│   └── model_config.json
│
├── outputs/
│
├── app.py
├── model.py
├── requirements.txt
├── START.bat
└── README.md
```

### Important Files

- `app.py`: Main Streamlit web application.
- `model.py`: Model initialization, checkpoint loading, and inference logic.
- `models/`: Contains the trained model checkpoint and configuration.
- `outputs/`: Stores generated outputs, if enabled by the application.
- `requirements.txt`: Python dependency list.
- `START.bat`: Windows one-click application launcher.
- `README.md`: Project documentation.

---

# 6. Installation and Setup

Follow these steps to set up the project on a Windows machine.

### Step 1: Install Python

Install Python 3.10 from:

https://www.python.org/downloads/

Verify the installation:

```bash
python --version
```

### Step 2: Open the Project Directory

Open PowerShell or the VS Code terminal inside the project directory:

```bash
cd "D:\Program Files\brain-tumor-segmentation"
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

```powershell
.venv\Scripts\activate
```

### Step 5: Install PyTorch

For NVIDIA GPU acceleration, install a PyTorch build compatible with the installed NVIDIA driver and supported CUDA runtime.

Official installation instructions:

https://pytorch.org/get-started/locally/

For CPU-only installation, use the CPU build provided by the official PyTorch installation selector.

### Step 6: Install Project Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 7. Requirements File

The following packages are required by the application:

```text
streamlit
torch
torchvision
segmentation-models-pytorch
numpy
pandas
matplotlib
Pillow
```

PyTorch installation should be selected separately according to the target machine's CPU or CUDA configuration before installing the remaining requirements.

---

# 8. Running the Application

The project includes a Windows launcher for one-click execution.

### Recommended Method: START.bat

1. Open the project folder.
2. Locate `START.bat`.
3. Double-click the file.
4. Wait for the application server to start.
5. Open the browser at the displayed local address.

Default application address:

```text
http://localhost:8501
```

If the default port is already occupied, the launcher may use another configured port.

### Manual Execution

Alternatively, run the application manually:

```powershell
.venv\Scripts\activate
python -m streamlit run app.py
```

Keep the terminal open while using the application.

To stop the server, press:

```text
CTRL + C
```

---

# 9. How to Use NeuroSeg AI

### Step 1: Launch the Platform

Run `START.bat` or start the application manually.

### Step 2: Upload an MRI Scan

Use the upload section to select a supported brain MRI image.

### Step 3: Preview the Image

Review the uploaded MRI scan and its image information.

### Step 4: Run Segmentation

Start the inference process using the pretrained U-Net model.

### Step 5: Review Results

Examine the predicted segmentation mask and its overlay on the original MRI image.

### Step 6: Export Results

Download the generated segmentation output if the export functionality is enabled in the application.

---

# 10. Model Inference Pipeline

The application follows this general workflow:

1. Receive an uploaded MRI image.
2. Convert the image to grayscale.
3. Resize the image to 256 × 256.
4. Convert the image into a PyTorch tensor.
5. Pass the tensor to the pretrained U-Net model.
6. Apply the sigmoid activation to model logits.
7. Apply the prediction threshold of 0.60.
8. Generate a binary segmentation mask.
9. Visualize the predicted mask and overlay.

The exact preprocessing and postprocessing operations are implemented in the application code.

---

# 11. Troubleshooting

### Problem: Model Checkpoint Not Found

Verify that the following file exists:

```text
models/brain_tumor_pretrained_unet_resnet34.pth
```

### Problem: CUDA Not Available

Check CUDA availability:

```python
import torch

print(torch.cuda.is_available())
```

If CUDA is unavailable, verify the NVIDIA driver and PyTorch installation. A compatible CPU configuration may be used if supported by the application.

### Problem: Port 8501 Is Already in Use

Close the previously running Streamlit process or use an available port.

### Problem: Missing Python Package

Activate the virtual environment and run:

```bash
pip install -r requirements.txt
```

### Problem: Application Does Not Start

Check the terminal output for Python errors, missing files, incorrect paths, or dependency conflicts.

---

# 12. Dataset Information

The model was developed using the LGG MRI Segmentation Dataset.

Dataset source:

https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

The dataset contains brain MRI images and corresponding ground-truth segmentation masks.

The training workflow includes data preprocessing, patient-level splitting, model training, validation, and final test evaluation.

---

# 13. Limitations and Future Improvements

### Current Limitations

- The model was evaluated using a specific dataset and test distribution.
- Small tumor regions may be difficult to segment accurately.
- False-positive predictions may occur on images without tumor annotations.
- External clinical validation has not been established.
- Model predictions should be reviewed by qualified medical professionals.

### Future Improvements

- Evaluate additional segmentation architectures.
- Improve performance on empty-mask images.
- Investigate additional MRI datasets.
- Perform external validation.
- Improve application usability and visualization.
- Explore model explainability techniques.

---

# 14. Conclusion

NeuroSeg AI demonstrates the practical application of deep learning in medical image segmentation.

By integrating a pretrained U-Net model with a Streamlit interface, the project provides an interactive workflow for MRI image upload, tumor-region segmentation, and visualization.

The platform connects model development and evaluation with a practical demonstration interface, making the experimental results easier to inspect and understand.

---

**Project Name:** NeuroSeg AI  
**Domain:** Artificial Intelligence | Deep Learning | Medical Image Segmentation  
**Primary Model:** U-Net with ResNet34 Encoder  
**Interface:** Streamlit  
**Language:** Python 3.10