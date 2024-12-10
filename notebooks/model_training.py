from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

def train_model(data_dir, model_path):
    """
    Entrena un modelo CNN y lo guarda en model_path.
    """
    # Cargar y preparar los datos aquí...
    # Crear modelo
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Cambia según el número de categorías
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Entrenar modelo
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save(model_path)
    print(f"Modelo guardado en {model_path}.")

def load_model(model_path):
    """
    Carga un modelo entrenado desde model_path.
    """
    return load_model(model_path)