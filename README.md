# Forecasting of Wind Turbine Power Generation
> Three Machine Learning and Statistics Approaches to predicting Actual Power Output of a Wind Turbine

Project Start: 23-May-2022

Original Dataset from Kaggle: [Wind Power Forecasting](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting)

## Files
1. Turbine_Data.csv: Dataset from Kaggle; the only modification is that the first column has been named for reference in Pandas
2. windpower.ipynb: Jupyter Notebook consisting of all three approaches
3. lng-wind.py: PyTorch Lighting-based LSTM/GRU forecasting model and DataModule. Also contains sklearn-based SVR for performance comparisson
4. plt.png: Saved figure comparing LSTM, SVR, and true outputs
5. sarimax-wind.py: Application of SARIMAX function(s) to forecast on the same dataset

## Getting Started
This project was created using Conda miniforge, as I constantly switch between ARM & x86 as well as Mac/Windows/Linux

1. Setup a conda environment using the provided yml file
2. Ensure that you have the csv and program file(s) in the same directory, or adjust it to fit

At this point, it should run relatively smoothly. Some issues do occur with Jupyter and multiple workers, so that value has been set to 0 in the ipynb file. Otherwise, it will automatically select a up to 80% of the available CPU cores depending on the task. 
* Note: Apple Silicon-based machines seem to complain about not utilizing all of the available CPU cores and Windows seems to crash (something, usually Python) when all cores are used.

## Results
*Error calculations taken after denormalizing the data for equal comparison*

| Model Type | Performance | MSE | rMSE | MAE |
| ---------- | ----------- | --- | ---- | --- |
| LSTM/GRU | Roughly equal to each other | >140,000 | 410~420 | ~330 |
| SVR | Small improvement | >140,000 | ~400 | ~300 |
| ARIMA | Major improvement but horizontal output | ~47,000 | ~217 | N/A |
| SARIMAX | Major improvement but low variability | ~67,300 | ~260 | ~205 |

Overall, a successful first foray into Machine Learning and ML-based Regression
