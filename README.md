# House-Price-Prediction

This project builds a neural network model to predict housing prices in King County, Washington (USA). The dataset contains house sale prices and features such as square footage, number of bedrooms, bathrooms, location, renovation history, and more.

The workflow includes data analysis, visualization, preprocessing, deep learning modeling, and evaluation.

Project Structure
house-price-prediction/
│
├── kc_house_data.csv         
├── house_price_model.h5      
├── house_price_prediction.py 
└── README.md                 

Features

Data Exploration and Visualization

Distribution plots of house prices and bedrooms

Correlation matrix heatmap

Scatterplots of price vs sqft_living and price vs longitude

Data Preprocessing

Handling missing values (if any)

Feature engineering (extracting year/month from date)

Dropping irrelevant columns (id, date, zipcode)

Normalizing features with MinMaxScaler

Neural Network Model (TensorFlow/Keras)

Sequential model with multiple dense layers

Early stopping to prevent overfitting

Optimizer: Adam | Loss: Mean Squared Error (MSE)

Model Evaluation

Loss curve visualization

Metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), Explained Variance

Prediction vs Actual plot

Price Prediction

Predicts price for a new single house instance

Scales input features before prediction

Example Visualizations

House Price Distribution

Bedrooms Count Distribution

Correlation Heatmap

Prediction vs Actual Scatterplot

Installation

Clone the repository and install dependencies:

git clone https://github.com/morgankhweku/House-Price-Prediction.git
cd house-price-prediction
pip install -r requirements.txt


requirements.txt should include:

pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow

Usage

Run the main script:

python house_price_prediction.py

Model Performance

Mean Squared Error (MSE): ~

Mean Absolute Error (MAE): ~

Explained Variance Score: ~

(Values depend on training run)

Future Improvements

Try log-transforming the target variable (price) for better stability.

Use feature selection to reduce noise.

Experiment with other models: Random Forest, XGBoost, or Gradient Boosting.

Hyperparameter tuning with Keras Tuner or Optuna.

Deploy model via Flask/FastAPI or as a web app.
Dataset
The dataset is from Kaggle’s King County House Sales dataset:
https://www.kaggle.com/harlfoxem/housesalesprediction
The dataset is from Kaggle’s King County House Sales dataset:
https://www.kaggle.com/harlfoxem/housesalesprediction
