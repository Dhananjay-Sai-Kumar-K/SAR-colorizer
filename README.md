# 🎨 SAR Image Colorization Project 📡

This project focuses on colorizing Synthetic Aperture Radar (SAR) images using deep learning techniques. It includes both the training scripts for building the colorization model and a Streamlit application for easy deployment and testing.

## 📂 Project Structure

## 🛠️ Setup

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd SAR-Colorization
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    source venv/bin/activate # On macOS/Linux
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Training the Model

The training scripts are located in the `training_scripts/` directory.

### 🧠 U-Net Model

To train the U-Net model, run the following command:

```bash
python training_scripts/training_unet.py

python training_scripts/training_vgg.py
