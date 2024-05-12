import shutil
import os

class DataProcessor:
    """
    Class to process data by splitting it into train, validation, and test sets.
    """
    def __init__(self, source_path, save_path, no_of_images):
        """
        Initialize DataProcessor object.

        Args:
            source_path (str): Path to the source data directory.
            save_path (str): Path to save the processed data.
            no_of_images (int): Total number of images in the dataset.
        """
        self.source_path = source_path
        self.save_path = save_path
        self.no_of_images = no_of_images

    def create_directory_if_not_exists(self, directory):
        """
        Create a directory if it doesn't exist.

        Args:
            directory (str): Directory path to create.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def process_data(self):
        """
        Process the data by splitting it into train, validation, and test sets.
        """
        train_threshold = int(0.7 * self.no_of_images)
        val_threshold = int(0.2 * self.no_of_images)

        directories = [
            os.path.join(self.save_path, "train", "images"),
            os.path.join(self.save_path, "train", "labels"),
            os.path.join(self.save_path, "val", "images"),
            os.path.join(self.save_path, "val", "labels"),
            os.path.join(self.save_path, "test", "images"),
            os.path.join(self.save_path, "test", "labels")
        ]

        for directory in directories:
            self.create_directory_if_not_exists(directory)

        images_list = os.listdir(os.path.join(self.source_path, "images"))

        for i, image_name in enumerate(images_list):
            source_image_path = os.path.join(self.source_path, "images", image_name)
            text_file_name = image_name[:-4] + ".txt"
            source_save_dir = os.path.join(self.source_path, "labels2", text_file_name)

            if i < train_threshold:
                dest_image_path = os.path.join(self.save_path, "train", "images", image_name)
                dest_save_dir = os.path.join(self.save_path, "train", "labels", text_file_name)
            elif i < train_threshold + val_threshold:
                dest_image_path = os.path.join(self.save_path, "val", "images", image_name)
                dest_save_dir = os.path.join(self.save_path, "val", "labels", text_file_name)
            else:
                dest_image_path = os.path.join(self.save_path, "test", "images", image_name)
                dest_save_dir = os.path.join(self.save_path, "test", "labels", text_file_name)

            shutil.copy(source_image_path, dest_image_path)
            shutil.copy(source_save_dir, dest_save_dir)

if __name__ == "__main__":
    no_of_images = 500
    source_path = "/mnt/c/MatriceAI/Deep_Fashion_Praneeth/data/"
    save_path = "/mnt/c/MatriceAI/Deep_Fashion_Praneeth/data/yolov9_format"

    data_processor = DataProcessor(source_path, save_path, no_of_images)
    data_processor.process_data()
