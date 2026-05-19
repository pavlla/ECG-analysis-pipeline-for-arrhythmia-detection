import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from config import (
    RANDOM_STATE,
    BATCH_SIZE,
    EPOCHS,
    CLASS_LABELS,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
)

from model import build_model


def split_data(X_segments, X_features, y):
    X_segments_train, X_segments_temp, X_features_train, X_features_temp, y_train, y_temp = train_test_split(
        X_segments,
        X_features,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_segments_val, X_segments_test, X_features_val, X_features_test, y_val, y_test = train_test_split(
        X_segments_temp,
        X_features_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return (
        X_segments_train,
        X_segments_val,
        X_segments_test,
        X_features_train,
        X_features_val,
        X_features_test,
        y_train,
        y_val,
        y_test,
    )


def scale_features(X_features_train, X_features_val, X_features_test):
    scaler = StandardScaler()

    X_features_train = scaler.fit_transform(X_features_train)
    X_features_val = scaler.transform(X_features_val)
    X_features_test = scaler.transform(X_features_test)

    return X_features_train, X_features_val, X_features_test, scaler


def train_model(X_segments, X_features, y):
    (
        X_segments_train,
        X_segments_val,
        X_segments_test,
        X_features_train,
        X_features_val,
        X_features_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(X_segments, X_features, y)

    X_features_train, X_features_val, X_features_test, scaler = scale_features(
        X_features_train,
        X_features_val,
        X_features_test,
    )

    model = build_model(
        segment_length=X_segments_train.shape[1],
        num_features=X_features_train.shape[1],
        num_classes=len(np.unique(y)),
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        min_delta=0.001,
    )

    history = model.fit(
        [X_segments_train, X_features_train],
        y_train,
        validation_data=([X_segments_val, X_features_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[lr_scheduler, early_stopping],
    )

    test_loss, test_acc = model.evaluate([X_segments_test, X_features_test], y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Model saved at: {MODEL_PATH}")
    print(f"Scaler saved at: {SCALER_PATH}")

    evaluate_model(
        model,
        history,
        X_segments_test,
        X_features_test,
        y_test,
        y_train,
    )

    return model, scaler, history


def evaluate_model(model, history, X_segments_test, X_features_test, y_test, y_train):
    y_pred_prob = model.predict([X_segments_test, X_features_test])
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))

    plot_roc_curve(y_test, y_pred_prob, y_train)
    plot_confusion_matrix(y_test, y_pred, y_train)
    plot_learning_curves(history)


def plot_roc_curve(y_test, y_pred_prob, y_train):
    y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

    plt.figure(figsize=(10, 6))

    for i, class_name in enumerate(CLASS_LABELS):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_name} AUC = {roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, y_train):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_LABELS,
    )

    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
