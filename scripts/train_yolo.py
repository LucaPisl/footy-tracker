import os

import torch
from ultralytics import YOLO


def main():
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 0:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load a pre-trained model (starting with yolov8n.pt as requested)
    # We can also use yolov11n.pt if preferred, but yolov8n.pt is a solid baseline.
    model = YOLO('yolov8n.pt')

    # Train the model
    # We use 50 epochs as suggested.
    # imgsz 640 is standard.
    results = model.train(
        data='configs/soccernet_data.yaml',
        epochs=50,
        imgsz=640,
        device=device,
        project='outputs/train',
        name='soccernet_player_detector',
        exist_ok=True
    )

    print("Training completed.")
    print(f"Best model saved at: {os.path.join('outputs/train', 'soccernet_player_detector', 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
