import os
from notebooks.data_preprocessing import preprocess_data # type: ignore
from model_training import train_model, load_model # type: ignore
from recommend import recommend_items # type: ignore
from visualization import visualize_recommendations # type: ignore

# Configuración
RAW_DATA_DIR = "./data/raw/"
PROCESSED_DATA_DIR = "./data/processed/"
MODEL_PATH = "./models/outfit_classifier.h5"

def main():
    """
    Función principal para ejecutar el flujo completo del proyecto.
    """
    # Paso 1: Preprocesamiento de Datos
    if not os.path.exists(PROCESSED_DATA_DIR):
        print("Preprocesando datos...")
        preprocess_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    else:
        print("Datos preprocesados ya disponibles.")

    # Paso 2: Entrenamiento del Modelo
    if not os.path.exists(MODEL_PATH):
        print("Entrenando el modelo...")
        train_model(PROCESSED_DATA_DIR, MODEL_PATH)
    else:
        print("Modelo ya entrenado disponible.")

    # Paso 3: Generación de Recomendaciones
    print("Cargando modelo y generando recomendaciones...")
    model = load_model(MODEL_PATH)
    recommendations = recommend_items(model, PROCESSED_DATA_DIR)

    # Paso 4: Visualización
    print("Mostrando recomendaciones...")
    visualize_recommendations(recommendations)

if __name__ == "__main__":
    main()