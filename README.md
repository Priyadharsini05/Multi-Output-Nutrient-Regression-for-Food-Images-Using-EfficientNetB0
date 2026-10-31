# Smart Nutrition | Image-Based Nutritional Analysis & Personalized Recommendations

**Smart Nutrition** is a deep learning–based food analysis system that estimates the nutritional composition of food images using transfer learning and multi-output regression. It combines computer vision, nutritional data preprocessing, and profile-based recommendation logic to help users evaluate food in relation to selected nutritional requirements.

The system uses **EfficientNetB0 pretrained on ImageNet** as its feature extraction backbone and predicts eight nutritional attributes from a single food image. Predicted values are then converted back to their original scales and evaluated against user-defined nutritional thresholds to generate recommendations and a nutritional suitability score.

## Key Features

- **Image-Based Nutrient Estimation:** Predicts eight nutritional attributes directly from food images.
- **Transfer Learning:** Uses pretrained EfficientNetB0 to extract visual representations of food.
- **Multi-Output Regression:** Predicts multiple continuous nutritional values through a shared neural network.
- **Data Augmentation:** Applies random rotations, translations, zooming, horizontal flipping, and brightness adjustments during training.
- **Nutrient Normalization:** Independently scales target variables using `MinMaxScaler` to a range of 0–100.
- **Inverse Transformation:** Converts normalized model predictions back to the original nutrient units.
- **Personalized Recommendations:** Evaluates predicted nutrient values against thresholds for user-selected nutritional priorities.
- **Training Optimization:** Uses early stopping and model checkpointing to retain a better-performing model.
- **Model Persistence:** Saves the trained model and nutrient scalers for subsequent inference.

## System Architecture

```text
                    FOOD IMAGE
                         |
                         v
              Image Preprocessing
             Resize to 224 x 224
              RGB Conversion
               Pixel Scaling
                         |
                         v
              EfficientNetB0
           ImageNet-Pretrained CNN
             Frozen Feature Layers
                         |
                         v
             Global Average Pooling
                         |
                         v
                Dense Layer
                 512 Neurons
                 ReLU Activation
                         |
                         v
                  Dropout (0.3)
                         |
                         v
             Linear Output Layer
                 8 Predictions
                         |
                         v
              Inverse Normalization
                         |
                         v
              Nutrient Predictions
                         |
                         v
             User Profile Evaluation
                         |
                         v
          Recommendations + Score
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| CNN Architecture | EfficientNetB0 |
| Transfer Learning | ImageNet-pretrained weights |
| Image Processing | Pillow (PIL), NumPy |
| Data Processing | Pandas, NumPy |
| Target Scaling | Scikit-learn `MinMaxScaler` |
| Model Persistence | Keras model serialization (`.h5`) |
| Scaler Persistence | Joblib |
| Training Callbacks | `EarlyStopping`, `ModelCheckpoint` |
| Data Augmentation | Keras `ImageDataGenerator` |

## Model Design

### 1. Image Preprocessing

Each food image is processed before being passed to the neural network.

The preprocessing pipeline includes:

- Loading images using Pillow.
- Converting images to RGB format.
- Resizing images to `224 × 224` pixels.
- Converting image data into NumPy arrays.
- Normalizing pixel values by dividing by `255.0`.

The resulting input tensor has the shape:

```text
(batch_size, 224, 224, 3)
```

Invalid or unreadable image files are skipped during dataset loading, allowing the remaining valid samples to be processed.

### 2. Nutritional Target Preparation

The model predicts the following eight nutritional attributes:

| Attribute | Target Column | Description |
|---|---|---|
| Carbohydrates | `carbs` | Estimated carbohydrate content |
| Fat | `fat` | Estimated total fat content |
| Fibre | `fibre` | Estimated dietary fibre content |
| Energy | `kcal` | Estimated energy content |
| Protein | `protein` | Estimated protein content |
| Salt | `salt` | Estimated salt content |
| Saturated Fat | `saturates` | Estimated saturated fat content |
| Sugars | `sugars` | Estimated sugar content |

Each target column is independently normalized using a separate `MinMaxScaler`:

```python
scaler = MinMaxScaler(feature_range=(0, 100))
nutrition_df[col + "_scaled"] = scaler.fit_transform(
    nutrition_df[[col]]
)
```

The fitted scalers are serialized using Joblib and reused during inference to reverse the normalization applied to the model's predictions.

**Why scale the targets?**

The eight output variables can have significantly different numerical ranges. Scaling them to a common interval helps prevent differences in magnitude from dominating the regression loss solely because of their numerical scale.

### 3. Transfer Learning with EfficientNetB0

The model uses EfficientNetB0 pretrained on ImageNet as its visual feature extractor.

```python
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False
```

The original classification head is removed, and the pretrained convolutional layers are frozen during training.

A custom regression head is attached to the backbone:

```python
x = layers.GlobalAveragePooling2D()(base_model.output)

x = layers.Dense(
    512,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

output = layers.Dense(
    8,
    activation="linear",
    name="nutrients"
)(x)

model = models.Model(
    inputs=base_model.input,
    outputs=output
)
```

**Architecture details:**

- **Backbone:** EfficientNetB0
- **Pretrained weights:** ImageNet
- **Feature aggregation:** Global Average Pooling
- **Regression head:** Dense layer with 512 units
- **Regularization:** Dropout with a rate of 0.3
- **Output layer:** Eight linear neurons
- **Optimization:** Adam
- **Loss function:** Mean Squared Error (MSE)
- **Evaluation metric:** Mean Absolute Error (MAE)

The linear output layer allows the model to estimate continuous values rather than assign food images to discrete classes.

### 4. Data Augmentation

Training images undergo randomized transformations to expose the model to variations in image orientation, framing, and brightness.

```python
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
```

The validation generator does not apply these transformations.

Training configuration:

| Parameter | Value |
|---|---|
| Input resolution | 224 × 224 |
| Batch size | 16 |
| Maximum epochs | 50 |
| Validation split | 20% |
| Random seed | 42 |
| Dropout rate | 0.3 |
| Early stopping patience | 5 epochs |
| Target range | 0–100 |

The dataset is divided into training and validation sets using `train_test_split`.

### 5. Training Optimization and Checkpointing

The model is compiled with mean squared error as the optimization objective and mean absolute error as an additional evaluation metric.

```python
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)
```

Two callbacks are configured:

- **EarlyStopping:** Monitors validation loss and stops training after five consecutive epochs without improvement. The best observed weights are restored.
- **ModelCheckpoint:** Saves the best model according to validation loss.

These callbacks help control unnecessary training and preserve a model with better validation performance.

## Nutrient Prediction Pipeline

The inference pipeline accepts a food image, generates predictions, and converts the normalized output into the original target scales.

### Image preprocessing

```python
def preprocess_image(img_path):
    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
```

### Prediction and inverse scaling

```python
def predict_nutrients(img_path):
    img_array = preprocess_image(img_path)

    pred_scaled = model.predict(
        img_array,
        verbose=0
    ).flatten()

    pred_original = []

    for i, nutrient in enumerate(nutrient_columns):
        scaler = scalers[nutrient]

        value = scaler.inverse_transform(
            [[pred_scaled[i]]]
        )[0][0]

        pred_original.append(round(value, 2))

    return dict(zip(nutrient_columns, pred_original))
```

The model returns eight normalized predictions. Each prediction is passed through its corresponding fitted scaler to obtain a value in the original target scale.

**Important:** The dataset's units, serving sizes, and nutritional measurement conventions determine how these values should be interpreted. The model does not independently identify portion size or verify the nutritional composition of a meal.

## Personalized Nutritional Recommendations

The recommendation module compares predicted nutrient values against predefined thresholds for the nutritional priorities selected in a user profile.

Example thresholds:

```python
thresholds = {
    "iron": 5,
    "vitamin_b": 0.5,
    "calcium": 200,
    "vitamin_d": 5,
    "vitamin_c": 20,
    "protein": 10,
    "fiber": 5,
    "potassium": 300
}
```

A user profile determines which attributes are evaluated:

```python
user_profile = {
    "protein": True,
    "iron": True,
    "vitamin_b": True
}
```

The recommendation function checks whether each selected attribute meets its configured threshold. It generates a warning for attributes that fall below their threshold and calculates a score based on the proportion of selected checks that pass.

```text
Score = (Number of Passing Checks / Total Selected Checks) × 100
```

If no nutritional priorities are selected, the score is returned as `None`.

**Current implementation note:** The prediction model outputs eight attributes: carbohydrates, fat, fibre, calories, protein, salt, saturated fat, and sugars. However, the recommendation module also includes iron, vitamin B, calcium, vitamin D, vitamin C, and potassium thresholds. These additional micronutrients are not predicted by the current model. Unless they are provided by another data source, their values default to zero during recommendation evaluation. Consequently, those checks cannot currently provide valid micronutrient estimates.

The score is a threshold-based suitability indicator, not a clinically validated nutrition score or a measure of overall dietary adequacy.

## Dataset and Data Handling

The training pipeline reads nutritional labels from a CSV file and associates each record with a food image.

Expected dataset structure:

```text
Smart_Nutrition/
├── nutrients/
│   ├── data.csv
│   └── images/
│       ├── food_image_1.jpg
│       ├── food_image_2.jpg
│       └── food_image_3.png
│
└── models/
    ├── food_model.h5
    └── nutrient_scalers.pkl
```

The CSV file must contain an `image` column and the eight numerical target columns:

```text
image, carbs, fat, fibre, kcal, protein, salt, saturates, sugars
```

Each `image` value must correspond to a filename inside the images directory.

The loader skips missing or unreadable images and reports the number of skipped records. Each successfully loaded image is paired with its corresponding normalized nutrient targets.

The reported training run loaded **1,249 valid image–nutrient pairs** after skipping nine invalid image records.

The dataset itself is not included in this repository structure by default. Provide your own appropriately licensed dataset and update the file paths before running the pipeline.

## Training Results

In the recorded execution, the model was trained with a maximum limit of 50 epochs and early stopping enabled.

The captured logs show the following results for the first epoch:

| Metric | Training | Validation |
|---|---:|---:|
| MSE Loss | 267.8987 | 200.4604 |
| MAE | 12.0099 | 10.3278 |

The logs subsequently show validation loss fluctuating around 199–202, with no clear sustained improvement in the displayed epochs.

These values are the metrics reported by the training process on the normalized target scale. They should not be interpreted as errors in grams, calories, or other original units.

The provided execution output does not establish the model's final generalization performance. A held-out test set and per-nutrient evaluation after inverse scaling would be needed to assess predictive accuracy more reliably.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- A compatible TensorFlow installation
- A labeled food image dataset with nutritional values

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Smart_Nutrition
```

Replace `<your-repository-url>` with the URL of your GitHub repository.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy pandas scikit-learn pillow joblib
```

For reproducible results, pin the package versions tested in your environment in a `requirements.txt` file.

## Configuration

The training script currently uses local Windows paths. Update these constants to match your environment:

```python
DATA_DIR = r"C:\path\to\Smart_Nutrition\nutrients\images"

NUTRITION_CSV = r"C:\path\to\Smart_Nutrition\nutrients\data.csv"

MODEL_SAVE_PATH = r"C:\path\to\Smart_Nutrition\models\food_model.h5"

SCALERS_PATH = r"C:\path\to\Smart_Nutrition\models\nutrient_scalers.pkl"
```

Ensure the output directory exists or allow the training script to create it before saving the model and scalers.

## Running the Project

### 1. Train the model

Save the training code as:

```text
train_model_augmented.py
```

Run:

```bash
python train_model_augmented.py
```

The training pipeline will:

1. Load the nutritional dataset.
2. Fit and save the nutrient scalers.
3. Load and preprocess valid images.
4. Split the dataset into training and validation subsets.
5. Build the EfficientNetB0-based regression model.
6. Train the regression head with augmented images.
7. Apply early stopping and checkpointing.
8. Save the trained model.

### 2. Run nutrient prediction

Save the inference code as:

```text
predict_nutrients.py
```

Update the model, scaler, and test-image paths in the script.

Run:

```bash
python predict_nutrients.py
```

The script loads the saved model and scalers, processes the selected image, and prints the predicted nutritional attributes.

Example output format:

```text
Predicted Nutrients: {
    'carbs': 36.7,
    'fat': 14.52,
    'fibre': 6.88,
    'kcal': 355.3,
    'protein': 17.19,
    'salt': 0.72,
    'saturates': 4.75,
    'sugars': 13.22
}
```

These values illustrate the output format from the recorded execution. They are example predictions for one image, not guaranteed results for other images.

## Model Artifacts

| Artifact | Purpose |
|---|---|
| `food_model.h5` | Serialized EfficientNetB0-based multi-output regression model |
| `nutrient_scalers.pkl` | Fitted `MinMaxScaler` instances used to reverse target normalization |

Keep the model and scaler artifacts together. The inference pipeline relies on the same fitted scalers used to transform the training targets.

For new training runs, save the model and its scalers as a matched pair so that the prediction pipeline uses consistent preprocessing and target transformations.

## Current Limitations

- **Visual estimation is approximate:** Nutrient content cannot be reliably determined from appearance alone, particularly when portion size, ingredients, and cooking methods are unknown.
- **Limited training data:** The recorded run uses 1,249 valid samples. Performance may be constrained by dataset size, image diversity, and label quality.
- **Micronutrient predictions are unavailable:** Iron, vitamin B, calcium, vitamin D, vitamin C, and potassium are not outputs of the current regression model.
- **No independent test evaluation is shown:** Training and validation logs alone are insufficient to establish real-world generalization.
- **Potential label ambiguity:** The reliability of predictions depends on how nutritional values were measured, normalized, and associated with images.
- **Frozen feature extractor:** The EfficientNetB0 backbone remains frozen in the supplied training script. Fine-tuning may improve performance, but would require experimentation.
- **Thresholds require validation:** Recommendation thresholds are manually configured and should not be interpreted as personalized medical guidance.
- **Training data leakage requires attention:** The supplied code fits scalers before splitting the dataset. For rigorous evaluation, fit preprocessing objects only on the training partition and apply them unchanged to validation and test data.

## Future Improvements

- Add a dedicated held-out test set and report per-nutrient MAE and RMSE after inverse scaling.
- Implement data leakage prevention by splitting the records before fitting target scalers.
- Compare EfficientNetB0 with alternative CNN backbones and fine-tuned transfer learning configurations.
- Evaluate performance across food categories and nutritional value ranges.
- Integrate a reliable source of micronutrient information to support the existing recommendation thresholds.
- Add portion-size estimation or serving-size input to improve nutritional estimates.
- Introduce configurable thresholds with clear units and documented sources.
- Develop an API or user interface for image uploads and inference.
- Export the model in a deployment-friendly format and benchmark inference latency.
- Add automated data validation and tests for preprocessing, prediction, and recommendation logic.

## Disclaimer

Smart Nutrition is an experimental machine learning project intended for educational and research purposes. Its predictions are estimates and should not replace verified nutritional labels, professional dietary assessment, or medical advice.

## Author

**Priyadharsini M.**

B.Tech — Computer Science and Engineering  
Specialization: Artificial Intelligence and Data Analytics

Interested in machine learning, deep learning, computer vision, and applied AI systems.
