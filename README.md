# Intelligent Face Recognition System using Siamese Networks

## Overview
This project implements a real-time identity verification system using a Siamese Neural Network trained on the Labeled Faces in the Wild (LFW) dataset. The system detects faces, compares them against known identities using similarity scores, and displays results through a Flask-based web application. Key features include:

- **Real-Time Webcam Recognition**: Detects and identifies faces in live video streams.
- **Image Pair Comparison**: Compares two uploaded images to determine if they depict the same person.
- **Identity Registration**: Allows users to register new faces with names for recognition.
- **Performance**: Achieves a training loss of 0.07 (10 epochs) and a test accuracy of 78.30%.

The project is developed as part of the ADL course (Semester 4) and includes a Flask web app, training/testing scripts, and an inference report (`inference.pdf`).

## Project Structure
```
mini-project/
├── dataset/
│   ├── lfw-deepfunneled/
│   │   ├── lfw-deepfunneled/lfw-deepfunneled/
│   │   │   ├── Abdullah_Gul/
│   │   │   ├── AJ_Lamas/
│   │   │   ├── ... (person folders)
│   │   ├── lfw_allnames.csv
│   │   ├── matchpairsDevTest.csv
│   │   ├── mismatchpairsDevTest.csv
│   │   ├── pairs.csv
│   │   ├── ... (other CSVs)
│   ├── shape_predictor_68_face_landmarks.dat
│   ├── test_pairs.csv
├── models/
│   ├── siamese_model.pth
├── static/
│   ├── haarcascade_frontalface_default.xml
│   ├── placeholder.jpg
│   ├── uploads/
├── templates/
│   ├── result.html
│   ├── upload.html
│   ├── webcam.html
├── src/
│   ├── app.py                # Flask web app
│   ├── create_test_pairs.py  # Generates test_pairs.csv
│   ├── demo.py              # Visualizes predictions
│   ├── download_dataset.py  # Downloads LFW dataset
│   ├── model.py             # Siamese Network definition
│   ├── test_dataset.py      # Tests LFWDataset
│   ├── test.py              # Evaluates model
│   ├── train.py             # Trains model
│   ├── utils.py             # Dataset utilities
├── demo_prediction.png      # Demo output
├── embeddings.pkl           # Known identities
├── inference.pdf            # Inference report
├── README.md                # This file
├── requirements.txt         # Dependencies
├── roc_curve.png            # ROC curve
├── sample_predictions.png   # Test predictions
├── venv/                    # Virtual environment
```

## Prerequisites
- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.8 or higher
- **Hardware**: Webcam (for real-time recognition), GPU/MPS (optional, for faster training)
- **Kaggle Account**: For downloading the LFW dataset (configure `kagglehub`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd mini-project
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include:
   - `torch==2.7.0`, `torchvision==0.22.0`
   - `opencv-python==4.11.0.86`
   - `flask==3.1.0`
   - `dlib==19.24.8` (for face alignment)
   - `kagglehub==0.3.12` (for dataset download)
   - See `requirements.txt` for the full list.

4. **Download LFW Dataset**:
   ```bash
   python3 src/download_dataset.py
   ```
   - Requires a Kaggle account. Set up `kagglehub`:
     ```bash
     export KAGGLE_USERNAME=your-username
     export KAGGLE_KEY=your-api-key
     ```
   - Outputs to `dataset/lfw-deepfunneled/`.

5. **Download Haar Cascade**:
   - Download `haarcascade_frontalface_default.xml` from [OpenCV's GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).
   - Place it in `static/`:
     ```bash
     mv haarcascade_frontalface_default.xml static/
     ```

## Usage
### 1. Train the Model
- Run the training script to train the Siamese Network on `pairs.csv`:
  ```bash
  python3 src/train.py
  ```
- Outputs `models/siamese_model.pth` after 10 epochs (final loss: 0.07).

### 2. Test the Model
- Generate test pairs from `matchpairsDevTest.csv` and `mismatchpairsDevTest.csv`:
  ```bash
  python3 src/create_test_pairs.py
  ```
- Run the test script to evaluate (accuracy: 78.30%):
  ```bash
  python3 src/test.py
  ```
- Outputs `roc_curve.png` and `sample_predictions.png`.

### 3. Run the Flask Web App
- Start the Flask server:
  ```bash
  python3 src/app.py
  ```
- Open `http://127.0.0.1:5000` in a browser.
- Features:
  - **Upload Images**: Compare two images (`/`) for same/different person.
  - **Register Identity**: Add a new face with a name (`/register`).
  - **Webcam Mode**: Real-time face recognition (`/webcam`).

### 4. Demo Predictions
- Visualize predictions for sample image pairs:
  ```bash
  python3 src/demo.py
  ```
- Outputs `demo_prediction.png`.

## Dataset
- **LFW Dataset**: Contains 13,000+ images of 5,749 individuals, organized in `dataset/lfw-deepfunneled/lfw-deepfunneled/lfw-deepfunneled/`.
- **Preprocessing**: Images are resized to 100x100 pixels, normalized, and optionally aligned using Dlib's 68-landmark predictor (`shape_predictor_68_face_landmarks.dat`).
- **Test Pairs**: Generated in `test_pairs.csv` for evaluation.

## Model
- **Architecture**: Siamese Network with three convolutional layers (64, 128, 256 filters) and fully connected layers producing 256-dimensional embeddings.
- **Loss**: Contrastive loss (margin=2.0).
- **Training**: 10 epochs, batch size 32, Adam optimizer (lr=0.001).
- **Performance**: Training loss of 0.07, test accuracy of 78.30%.

## Outputs
- `models/siamese_model.pth`: Trained model.
- `embeddings.pkl`: Known identity embeddings.
- `roc_curve.png`: ROC curve from testing.
- `sample_predictions.png`: Sample test predictions.
- `demo_prediction.png`: Demo visualization.
- `inference.pdf`: Project report.

## Troubleshooting
- **Dependency Issues**:
  - Ensure `dlib` compiles with `cmake` and `libpng`:
    ```bash
    brew install cmake libpng  # On macOS
    ```
  - If `torch` fails on MPS, use CPU by editing `src/*.py` to set `device='cpu'`.
- **Dataset Errors**:
  - Verify `dataset/lfw-deepfunneled/` contains person folders and CSVs.
  - Run `src/create_test_pairs.py` if `test_pairs.csv` is missing.
- **Flask App**:
  - Check `static/haarcascade_frontalface_default.xml` exists.
  - Ensure permissions:
    ```bash
    chmod -R 755 static/
    ```

## Contributing
- Fork the repository and submit pull requests for improvements.
- Report issues via GitHub Issues.
- Suggested enhancements:
  - Integrate MTCNN for better face detection.
  - Add video upload support.
  - Create a statistics dashboard in the Flask app.

## Acknowledgments
- **LFW Dataset**: Provided by the University of Massachusetts.
- **Libraries**: PyTorch, OpenCV, Flask, Dlib, and others listed in `requirements.txt`.
- **Author**: Md Zohaib
