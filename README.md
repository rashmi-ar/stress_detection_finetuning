# Stress Detection using finetuning

## Data Collection:

- An experiment with 25 participants, using an Empatica E4 wristband to record their physiological signals and determine their cognitive stress levels.
- The experimental design can be found on [GitHub](https://github.com/rashmi-ar/stress_detection_experiment "Stress-Detection-Experiment")

## Data pre-processing

- e4_raw_data\preprocessing_e4.ipynb
  - preprocessing raw E4 data
- pre-processed and segmented using a sliding window of the length of 30 seconds without overlap
- EDA, BVP, TEMP, HR and 3-axis ACC data from Empatica E4 is used for analysis

## Cross Validations

- Kfold (classifiers\kfold)
- Leave-One-Subject-Out (LOSO) (classifiers\loso)
- fine-tuning on LOSO (classifiers\loso)

## Models

- FCN
- ResNet
- Transformers
- LSTM

## Dashboard

- displays dynamic stress fluctuations by utilizing the insights gained from prediction outcomes upon user-specific data
- The application includes a stress meter, which enables users to visually understand their stress levels

![alt text](https://github.com/rashmi-ar/stress_detection_finetuning/blob/master/assets/stress_meter.png)

![alt text](https://github.com/rashmi-ar/stress_detection_finetuning/blob/master/assets/dashboard_graph.png)

## References

- Maciej Dzieżyc, Martin Gjoreski, Przemysław Kazienko, Stanisław Saganowski, and Matjaž Gams. Can we ditch feature engineering? end-to-end deep learning for affect recognition from physiological sensor data. Sensors, 20(22):6535, 2020

- Zhiguang Wang, Weizhong Yan, and Tim Oates. Time series classification from scratch with deep neural networks: A strong baseline. In 2017 International joint conference on neural networks (IJCNN), pages 1578–1585. IEEE, 2017

- Theodoros Ntakouris. https://keras.io/examples/timeseries/, 2021

- Jürgen Schmidhuber, Sepp Hochreiter, et al. Long short-term memory. Neural Comput, 9(8):1735–1780, 1997
