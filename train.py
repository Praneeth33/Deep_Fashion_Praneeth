import os
import yaml
from ultralytics import YOLO

class TrainYOLO:
    """
    Class to train YOLO model using specified hyperparameters and data configurations.
    """
    def __init__(self, model_path, hyp_config_path, data_config_path):
        """
        Initialize TrainYOLO object.

        Args:
            model_path (str): Path to the YOLO model file.
            hyp_config_path (str): Path to the hyperparameter configuration file.
            data_config_path (str): Path to the data configuration file.
        """
        self.model = YOLO(model_path)
        self.hyp_config_path = hyp_config_path
        self.data_config_path = data_config_path

    def load_hyp_config(self):
        """
        Load hyperparameter configuration from YAML file.

        Returns:
            dict: Hyperparameter configuration.
        """
        with open(self.hyp_config_path, "r") as f:
            return yaml.safe_load(f)

    def train_model(self):
        """
        Train the YOLO model using specified configurations.

        Returns:
            dict: Training results.
        """
        data = self.load_hyp_config()
        results = self.model.train(
            data=self.data_config_path,
            task=data['task'],
            epochs=data['epochs'],
            batch=data['batch'],
            imgsz=data['imgsz'],
            optimizer=data['optimizer'],
            device=data['device'],
            save_period=data['save_period'],
            workers=data['workers'],
            time=data['time'],
            pretrained=data['pretrained'],
            cache=data['cache']
        )
        return results

if __name__ == "__main__":
    model_path = 'weights/yolov9c-seg.pt'  # Path to YOLO model
    hyp_config_path = '/mnt/c/MatriceAI/Deep_Fashion_Praneeth/config/hyp.yaml'  # Path to hyperparameter config
    data_config_path = '/mnt/c/MatriceAI/Deep_Fashion_Praneeth/config/data.yaml'  # Path to data config
    trainer = TrainYOLO(model_path, hyp_config_path, data_config_path)
    results = trainer.train_model()
