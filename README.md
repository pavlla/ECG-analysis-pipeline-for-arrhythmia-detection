# ECG-analysis-pipeline-for-arrhythmia-detection

Intensive efforts are currently underway to enable deep space exploration. Long-duration
space missions pose a number of challenges for scientists and engineers, one of the key ones being the provision of continuous monitoring of the crew's vital parameters. In conditions of microgravity and exposure to ionizing radiation, astronauts' cardiovascular systems are subjected to significant stress. These specific conditions can lead to the development of cardiac arrhythmias, which in the long term may result in the formation of dangerous blood clots. Arrhythmia poses a serious threat not only in space. On Earth, it is one of the most common chronic cardiovascular diseases. Untreated arrhythmia can lead to strokes and further degradation of arterial vessels, which significantly lowers patients' quality of life and increases the risk of death. Therefore, monitoring arrhythmia incidents and precise assessment of their parameters are crucial for timely intervention—both using digital methods and direct clinical actions. The development of technology in the field of biomedical signal analysis, particularly electrocardiographic (ECG) signals, opens up new possibilities in remote monitoring and diagnosis of heart rhythm disorders. The use of advanced algorithms and machine learning methods allows for increased precision in arrhythmia detection, which has a direct impact on the effectiveness of treatment and prevention of cardiovascular diseases.


The aim of the project was to develop an ECG signal analysis pipeline capable of detecting sinus, atrial and ventricular arrhythmia. For this purpose, a multi-stage solution development and implementation process was designed, based on the hybrid BiLSTM network, ensuring effective and precise arrhythmia detection.

**This process includes several key steps:**

**•	Obtaining a database.**

**•	ECG signal pre-processing (including noise and artifact removal).**

**• Segmentation of the ECG signal into 5-second fragments.**

**•	Feature extractions.**

**•	Data augmentation for class N.**

**•	Designing a BiLSTM hybrid network model.**

**•	Model training and testing.**

**•	Model evaluation and inference.**

**•	Using a model to detect real-time arrhythmias.**



<img width="747" alt="Zrzut ekranu 2025-01-31 o 16 47 56" src="https://github.com/user-attachments/assets/f48d88ea-2eaf-4067-b603-d4db508410fe" />


# Dataset

The database used in the project was developed through a collaboration between the First
Affiliated Hospital of Nanjing Medical University and Southeastern University in China[https://doi.org/10.1007/s40846-020-00554-3]. The ECG data comes from a wireless wearable ECG monitor that has passed FDA certification. The results came from more than 200 arrhythmia sufferers, ranging in age from 18 to 82. All subjects were trained to use the wireless ECG monitor so that they could wear it from 24 hours to as long as 72 hours. The recordings obtained with the wearable device are 6-lead ECG recordings,digitized at 400 samples per second per channel, in 12-bit resolution, covering a frequency response of 0.05 Hz to 40 Hz. The arrhythmia database contains three categories: sinus arrhythmia, atrial arrhythmia, and ventricular arrhythmia. **It is the first program in the world to use this database to detect arrhythmias.**

# ECG signal pre-processing

To denoise ECG signals, wavelet transformation using the db5 wavelet and Butterworth filters: low-pass and high-pass were used. The signal was then divided into 5-second fragments, which at a sampling rate of 400 Hz corresponds to 2000 samples per segment.


<img width="827" alt="Zrzut ekranu 2025-01-31 o 17 41 52" src="https://github.com/user-attachments/assets/dae74ed0-c336-471d-b6bc-aa8004f877ab" />
<img width="828" alt="Zrzut ekranu 2025-01-31 o 17 43 19" src="https://github.com/user-attachments/assets/438060fc-a9d6-4d5c-bbf0-8731a2fe5fca" />
<img width="843" alt="Zrzut ekranu 2025-01-31 o 17 43 35" src="https://github.com/user-attachments/assets/e9b2d218-b755-4e3b-b953-d0a5d0cf0d7d" />
<img width="837" alt="Zrzut ekranu 2025-01-31 o 17 43 45" src="https://github.com/user-attachments/assets/beb42354-6ab7-4242-af87-276496ac7538" />

# Feature extractions

After denoising and segmenting the signal, 18 features were extracted (Table 3.2), which
can be categorized into four main groups: heart rate variability (HRV) features for tracking
time-domain intervals between QRS complexes (Min Heart Rate, Max Heart Rate, Average
Heart Rate, rMSSD, PRR50, PRR20, SDSD, SDRR), features based on Poincaré metrics (SD1,
SD2), morphological features (Average Peak Amplitude, PQ/QR/RS/ST Delays), and
frequency-domain features (Low Frequency, Mid Frequency, High Frequency).

# CNN-BiLSTM architecture with attention mechanism

A Bi-LSTM deep learning model was used, which processes sequences in both directions simultaneously, along with an attention mechanism, enabling more accurate classification of ECG signals. The first part of the model consists of convolutional layers (Conv1D), responsible for extracting features and patterns from the signal. These layers use filters of varying sizes, ReLU activation, and L2 regularization. BatchNormalization and MaxPooling1D are applied after each layer to stabilize the learning process and reduce the dimensionality of the data. The signals then pass through a Bi-LSTM layer, and the attention mechanism assigns weights to the important parts of the signal, improving selectivity in ECG analysis. After processing the signals, the data from the attention layer and additional features are combined using a Concatenate layer. They then pass through a Dense layer with BatchNormalization and dropout to prevent overfitting. The model ends with an output layer with a softmax activation function, calculating the probability of belonging to a given class. The Adam optimizer was used to adjust the learning rates during training.

<img width="1016" alt="Zrzut ekranu 2025-01-31 o 17 56 37" src="https://github.com/user-attachments/assets/6305e299-a5d3-4f80-825d-90f1373417f0" />


