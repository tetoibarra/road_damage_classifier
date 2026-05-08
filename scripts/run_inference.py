import argparse
import sys
import os

# Permite importar src/ desde cualquier ubicación
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predict import load_model, predict

def main():
    parser = argparse.ArgumentParser(description='Road Damage Classifier')
    parser.add_argument('--input', required=True, help='Ruta a la imagen (.jpg / .jpeg / .png)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: No se encontró la imagen '{args.input}'")
        sys.exit(1)

    print("🔄 Cargando modelo...")
    model = load_model()

    print(f"📸 Procesando: {args.input}")
    label, confidence = predict(args.input, model)

    print("\n========== RESULTADO ==========")
    print(f"Input:       {args.input}")
    print(f"Prediction:  {label}")
    print(f"Confidence:  {confidence:.2%}")
    print("================================")

if __name__ == '__main__':
    main()