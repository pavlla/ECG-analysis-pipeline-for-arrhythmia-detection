BASE_PATH = "An open-access arrhythmia database of wearable dynamic electrocardiogram.rar"
EXTRACT_PATH = "Extracted/"

SAMPLING_RATE = 400
SEGMENT_DURATION = 5
SEGMENT_LENGTH = SAMPLING_RATE * SEGMENT_DURATION

LABEL_MAPPING = {
    "N": 0,
    "A": 1,
    "V": 2,
}

CLASS_LABELS = ["N", "A", "V"]

MODEL_DIR = "saved_model"
MODEL_PATH = f"{MODEL_DIR}/arrhythmia_model.keras"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"

RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 50
