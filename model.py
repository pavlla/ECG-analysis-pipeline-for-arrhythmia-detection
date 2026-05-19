import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Bidirectional,
    LSTM,
    Reshape,
    Concatenate,
    Layer,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


def build_model(segment_length, num_features, num_classes):
    input_signal = Input(shape=(segment_length, 1), name="ecg_signal")

    x = Conv1D(32, 5, activation="relu", kernel_regularizer=l2(0.01))(input_signal)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 5, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Conv1D(128, 3, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Reshape((-1, 128))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.4))(x)
    x = Attention()(x)

    input_features = Input(shape=(num_features,), name="ecg_features")

    f = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(input_features)
    f = BatchNormalization()(f)
    f = Dropout(0.4)(f)

    merged = Concatenate()([x, f])

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[input_signal, input_features], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
