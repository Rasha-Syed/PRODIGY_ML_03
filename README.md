# PRODIGY_ML_03
# Task 03 : Cat vs Dog Image Classification with SVM

## Objective

Implement a Support Vector Machine (SVM) model to classify images of cats and dogs from the Kaggle dataset.

## Dataset

- **train.zip**: Contains training images of cats and dogs.
- **test.zip**: Contains test images of cats and dogs.
- The dataset is sourced from Kaggle.

## Steps Implemented

### 1. Data Loading and Exploration

- **Unzipping the Dataset**: Extracted images from `train.zip` and `test.zip` to access the dataset.
- **Exploration**: Reviewed the dataset structure and confirmed the presence of images categorized into `cats` and `dogs` folders.

### 2. Data Preprocessing

- **Image Loading**: Loaded images from the `train` and `test` directories.
- **Resizing and Grayscale Conversion**: Resized images to 128x128 pixels and converted them to grayscale.
- **Feature Extraction**: Extracted Histogram of Oriented Gradients (HOG) features from the grayscale images.

### 3. Model Building

- **Feature Scaling**: Applied `StandardScaler` to normalize the features.
- **Training and Validation Split**: Split the training data into training and validation sets using `train_test_split`.
- **SVM Model Training**:
  - **Parameter Tuning**: Used Grid Search with Cross-Validation (`GridSearchCV`) to find the optimal hyperparameters for the SVM model.
  - **Model Training**: Trained the SVM model using the best parameters.

### 4. Model Evaluation

- **Validation**: Evaluated the SVM model on the validation set.
- **Metrics**:
  - Calculated accuracy score.
  - Displayed a confusion matrix to visualize model performance.

### 5. Testing and Predictions

- **Test Data Preparation**: Loaded and preprocessed test images.
- **Model Prediction**: Used the trained model to predict labels for the test images.
- **Accuracy Calculation**: Evaluated accuracy on the test dataset (if true labels are available).

### 6. Visualization

- **Confusion Matrix**: Visualized model performance using a confusion matrix.
- **Image Classification**: Classified and displayed specific images with predicted labels.
- **Random Samples**: Displayed random images from the dataset with their true and predicted labels.


##  Model Evaluation Metrics

- **Validation Accuracy**: Accuracy score on the validation data.
- **Test Accuracy**: Accuracy score on the test data (if applicable).

## Files Included

- **train.zip**: Zip file containing training images.
- **test.zip**: Zip file containing test images.

## Usage

1. **Unzip the Dataset**: Run the `unzip_data()` function to extract images from `train.zip` and `test.zip`.
2. **Load and Preprocess Images**: Use `load_images_from_folder()` to load and preprocess images.
3. **Train the Model**: Execute the script to train the SVM model and optimize hyperparameters.
4. **Evaluate and Test**: Assess model performance on the validation and test datasets.
5. **Classify and Visualize**: Use `classify_and_display_image()` to classify and visualize specific images. Use `display_random_images()` to view random samples with true and predicted labels.


## Requirements

- `numpy`
- `scikit-learn`
- `opencv-python`
- `scikit-image`
- `matplotlib`


