# ECG-analysis-pipeline-for-arrhythmia-detection

Intensive efforts are currently underway to enable deep space exploration. Long-duration
space missions pose a number of challenges for scientists and engineers, one of the key ones
being the provision of continuous monitoring of the crew's vital parameters. In conditions of
microgravity and exposure to ionizing radiation, astronauts' cardiovascular systems are
subjected to significant stress. These specific conditions can lead to the development of cardiac
arrhythmias, which in the long term may result in the formation of dangerous blood clots.
Arrhythmia poses a serious threat not only in space. On Earth, it is one of the most common
chronic cardiovascular diseases. Untreated arrhythmia can lead to strokes and further
degradation of arterial vessels, which significantly lowers patients' quality of life and increases
the risk of death. Therefore, monitoring arrhythmia incidents and precise assessment of their
parameters are crucial for timely intervention—both using digital methods and direct clinical
actions. The development of technology in the field of biomedical signal analysis, particularly
electrocardiographic (ECG) signals, opens up new possibilities in remote monitoring and
diagnosis of heart rhythm disorders. The use of advanced algorithms and machine learning
methods allows for increased precision in arrhythmia detection, which has a direct impact on
the effectiveness of treatment and prevention of cardiovascular diseases.

The aim of the project was to develop an ECG signal analysis pipeline capable of detecting sinus, atrial and ventricular arrhythmia. For this purpose, a multi-stage solution development and implementation process was designed, based on the hybrid BiLSTM network, ensuring effective and precise arrhythmia detection.

This process includes several key steps: 

-Obtaining a database
-ECG signal pre-processing (including noise and artifact removal).
Feature extractions.
Data augmentation for class N.
Designing a BiLSTM hybrid network model.
Model testing.
Model evaluation and inference. 
Using a model to detect real-time arrhythmias.
<img width="747" alt="Zrzut ekranu 2025-01-31 o 16 47 56" src="https://github.com/user-attachments/assets/f48d88ea-2eaf-4067-b603-d4db508410fe" />
