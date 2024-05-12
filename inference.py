from ultralytics import YOLO
import time

def run_yolo_inference(model_weights_path, image_path):
    """
    Run YOLO object detection inference on an image.

    Args:
        model_weights_path (str): Path to the YOLO model weights file.
        image_path (str): Path to the image for inference.

    Returns:
        dict: Detection results.
    """
    start_time = time.time()
    model = YOLO(model_weights_path)

    # Perform inference
    results_original = model.predict(source=image_path, save=True)

    end_time = time.time()
    runtime = end_time - start_time
    print("Code runtime:", runtime, "seconds")

    return results_original

if __name__ == "__main__":
    model_weights_path = "weights/best.pt"  # Path to the trained model weights
    image_path = "/mnt/c/MatriceAI/Deep_Fashion_Praneeth/data/yolov9_format/test/images/000031.jpg"  # Path to the input image
    results = run_yolo_inference(model_weights_path, image_path)
