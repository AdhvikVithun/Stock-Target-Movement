Predictive Financial Analysis with Ensemble Models

** To view ouput , ppt and conference paper go to Ouputes file **

Overview:
This repository contains a comprehensive Python program for predictive financial analysis, focusing on stock price forecasting using a diverse set of machine learning models. The project employs time series analysis, deep learning, and ensemble techniques to enhance the accuracy of stock price predictions.


![Program Flow](https://github.com/AdhvikVithun/Stock-Target-Movement/assets/148479685/e38f26f6-eedb-46c9-8bf5-df8fee1a3ca3)
![Program flow 2](https://github.com/AdhvikVithun/Stock-Target-Movement/assets/148479685/0ee70b67-71df-416a-8f90-b15dd668aabb)


Table of Contents:
Introduction
Data Collection
Feature Engineering
Target Variable Setup
Exploratory Data Analysis
Scaling and Preprocessing
LSTM-based Autoencoder
XGBoost Regressor
LSTM Model
Ensemble Approach
Results and Evaluation
Conclusion


Introduction:
The program utilizes various machine learning models, including LSTM-based autoencoders, XGBoost regressors, and Support Vector Machines (SVM), to predict future stock price movements. The ensemble approach combines these models to achieve a more robust and accurate forecasting mechanism.

Data Collection:
Historical stock price data for the company 'META' is collected using the Yahoo Finance API. The data is stored in a CSV file for further analysis.

Feature Engineering:
Technical indicators such as Relative Strength Index (RSI) and Exponential Moving Averages (EMAs) are computed and added to the dataset for better model performance.

Target Variable Setup:
The target variable is defined as the difference between the adjusted closing price and the opening price of the stock. A binary classification ('TargetClass') is also created based on the direction of this difference.

Exploratory Data Analysis:
Visualizations are provided to explore the distribution and trends of the target variable, offering insights into the dataset.

Scaling and Preprocessing:
Data is scaled using Min-Max scaling, and the dataset is split using TimeSeriesSplit for training and testing purposes.

LSTM-based Autoencoder:
A deep learning model with LSTM layers is implemented to capture temporal patterns in the data, enhancing feature representations.

XGBoost Regressor:
An XGBoost regressor is trained using the encoded representations from the autoencoder, contributing to the predictive ensemble.

![LSTM AE + XG](https://github.com/AdhvikVithun/Stock-Target-Movement/assets/148479685/8fd74c04-7b8f-49fc-9cbd-8399d0081cf2)

LSTM Model:
A standalone LSTM model is created to further exploit temporal dependencies in the dataset.

Ensemble Approach:
The final predictions are generated by combining the outputs of the XGBoost regressor, SVM regressor, and the LSTM model, providing a more robust forecast.

![ensemble 1](https://github.com/AdhvikVithun/Stock-Target-Movement/assets/148479685/633d62a2-c7b9-476f-af24-8eba0c61caa9)

Results and Evaluation:
Evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are employed to assess the accuracy of individual models and the ensemble approach.

Conclusion:
The program showcases a holistic approach to stock price prediction, leveraging the strengths of various machine learning techniques. The ensemble model demonstrates improved accuracy, providing a valuable tool for financial analysts and investors.

Note: The provided code and analysis are for educational and experimental purposes. The predictive accuracy may vary based on data quality, market conditions, and other factors.



Just tried this model:
![Just Tried this Model](https://github.com/AdhvikVithun/Stock-Target-Movement/assets/148479685/a073a298-aedc-4e66-8225-854baa041e27)

