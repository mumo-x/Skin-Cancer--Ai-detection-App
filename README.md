# Skin Cancer Detection Web Application

This web application uses feature analysis to detect and classify potential skin cancer. It provides a user-friendly interface for entering skin lesion features and receiving predictions about potential skin cancer types.

## Features

- Enter and analyze skin lesion features (color, size, shape, texture, and evolution)
- Get predictions with confidence levels
- View alternative possible classifications
- Responsive and clean user interface
- Educational information about skin cancer types

## Dataset

The application uses the "Skin cancer DATA" dataset which includes the following categories:
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion

## Technical Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Model Architecture**: Convolutional Neural Network (CNN)

## Setup and Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. Clone or download this repository

2. Install the required dependencies using the setup script (recommended):
   ```
   python setup.py
   ```
   
   Alternatively, you can install dependencies manually:
   ```
   pip install -r requirements.txt
   ```

3. Verify your environment is set up correctly:
   ```
   python check_env.py
   ```

4. Train the model (this may take some time depending on your hardware):
   ```
   python model.py
   ```

5. Start the web application:
   ```
   python app.py
   ```

6. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

### Troubleshooting

If you encounter errors related to NumPy or pandas compatibility:

1. Try using the setup script which installs packages in the correct order:
   ```
   python setup.py
   ```

2. If you still have issues, try manually uninstalling and reinstalling the packages:
   ```
   pip uninstall -y numpy pandas
   pip install numpy==1.23.5
   pip install pandas==1.5.3
   pip install -r requirements.txt
   ```

## Usage

1. On the home page, enter the skin lesion features:
   - Color (brown, black, blue/gray, red, white, multi-colored)
   - Size in millimeters
   - Shape (symmetric, asymmetric, irregular borders, regular borders)
   - Texture (smooth, rough, scaly, ulcerated, crusty)
   - Whether the lesion has changed over time
2. Click "Analyze Features" to process the information
3. View the prediction results, including:
   - Primary prediction with confidence level
   - Alternative possible classifications
4. Click "New Analysis" to enter different features

## Important Notes

- This application is for educational purposes only and should not replace professional medical advice
- Always consult a dermatologist for proper diagnosis of skin conditions
- The model's predictions are based on training data and may not be 100% accurate

## Requirements File

A `requirements.txt` file is included with all necessary Python packages.

## Disclaimer

This tool is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.