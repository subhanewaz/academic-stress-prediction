# Academic Stress Level Prediction

A machine learning project that predicts a student's academic stress level (1–5) from survey responses about study environment, peer pressure, academic pressure from home, coping strategies and habits.
Built for CSE422 (Artificial Intelligence) at BRAC University.

## What's inside

- **Data preprocessing:** encoding categorical answers, mean imputation for missing values, one-hot encoding of coping strategies
- **Exploratory analysis:** histograms, correlation heatmap, scatter matrix, class distribution
- **Models:**
  - Linear Regression (R² score and mean squared error)
  - K-Nearest Neighbors classifier
  - Neural Network (Keras, two hidden layers, softmax over 5 stress levels)
  - K-Means clustering on peer pressure vs. academic pressure from home

## Results

- **Neural Network:** 62% accuracy in classifying stress levels (5 classes) on the test set

## Tech stack

Python • pandas • NumPy • scikit-learn • TensorFlow / Keras • Matplotlib • seaborn

## Run it

```bash
pip install -r requirements.txt
python academic_stress_prediction.py
```

## Dataset

The dataset (`academic_stress_level_dataset.csv`) is **not included** in this repository, as I am not permitted to share it publicly. To run the script, place a copy of the dataset in the same folder as `academic_stress_prediction.py`.

## My contribution

- Designed, trained and evaluated the Neural Network classifier (Keras), reaching 62% test accuracy
- Built the Linear Regression model and evaluated it with R² and mean squared error
- Evaluated the K-Nearest Neighbors classifier's accuracy
- Collaborated on exploratory data analysis: histograms, correlation heatmap and scatter matrix
- Co-developed K-Means clustering on peer pressure vs. academic pressure from home
