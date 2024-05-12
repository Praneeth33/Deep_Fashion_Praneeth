import json
import os


class COCO_reduce:
    """
    Class for reducing COCO dataset based on specific categories.
    """

    def __init__(self, path, annots_file, total_imgs, category_names):
        """
        Initialize COCO_reduce object.

        Args:
            path (str): Path to COCO dataset directory.
            annots_file (str): Filename of COCO annotations JSON file.
            total_imgs (int): Total number of images to keep per category.
            category_names (list): List of category names to keep.
        """
        self.path = path
        self.annots_file = annots_file
        self.total_imgs = total_imgs
        self.category_names = category_names
        self.no_category_names = len(category_names)
        self.annot_dict = {}

    def make_cat_list(self, data):
        """
        Make list of categories based on provided category names.

        Args:
            data (dict): COCO dataset annotations.

        Returns:
            list: List of categories matching category_names.
        """
        cat_list = []

        for categories in data["categories"]:
            if categories["name"] in self.category_names:
                cat_list.append(categories)

        return cat_list

    def give_cat_ids(self):
        """
        Get category IDs and initialize class count.

        Returns:
            tuple: Tuple containing category IDs and class count.
        """
        category_ids = []
        class_count = {}

        for i in self.annot_dict["categories"]:
            category_ids.append(i["id"])
            class_count[i["id"]] = 0

        return category_ids, class_count

    def make_annot_lists(self, data, category_ids, class_count):
        """
        Make list of annotations and count list.

        Args:
            data (dict): COCO dataset annotations.
            category_ids (list): List of category IDs to keep.
            class_count (dict): Dictionary containing class count.

        Returns:
            tuple: Tuple containing annotation list and count list.
        """
        annot_id = 0
        annot_list = []
        count_list = []
        img_count = 0

        for annotations in data["annotations"]:
            if img_count < 501:
                if annotations["category_id"] in category_ids and class_count[annotations["category_id"]] < self.total_imgs // self.no_category_names:
                    ann = annotations

                    annot_id += 1
                    ann["id"] = annot_id
                    annot_list.append(ann)

                    if annotations["image_id"] not in count_list:
                        count_list.append(annotations["image_id"])
                        class_count[annotations["category_id"]] += 1
                        img_count += 1
            elif img_count >= 501:
                break

        return annot_list, count_list

    def make_images_list(self, data, count_list):
        """
        Make list of images based on count list.

        Args:
            data (dict): COCO dataset annotations.
            count_list (list): List of image IDs to keep.

        Returns:
            list: List of images.
        """
        images_list = []
        for images in data["images"]:
            if images["file_name"][:-4] in count_list:
                images_list.append(images)

        return images_list

    def save_dict_to_json(self, dictionary, file_path):
        """
        Save dictionary to JSON file.

        Args:
            dictionary (dict): Dictionary to be saved.
            file_path (str): File path to save JSON.
        """
        with open(file_path, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4)

    def reduce_data(self, output_file):
        """
        Reduce COCO dataset based on provided categories.

        Args:
            output_file (str): Output file name for reduced dataset.
        """
        annots_loc = os.path.join(self.path, "annotations", self.annots_file)

        with open(annots_loc, "r") as f:
            data = json.load(f)

        self.annot_dict["categories"] = self.make_cat_list(data)
        category_ids, class_count = self.give_cat_ids()
        self.annot_dict["annotations"], count_list = self.make_annot_lists(data, category_ids, class_count)
        self.annot_dict["images"] = self.make_images_list(data, count_list)
        self.save_dict_to_json(self.annot_dict, output_file)


if __name__ == "__main__":
    path = "/mnt/c/Users/Praneeth Sai/Downloads/deep_fashion/deep_fashion"
    output_file = "test1.json"
    category_names = ["trousers", "short sleeve top"]
    total_imgs = 500

    reducer = COCO_reduce(path, "instances_train2024.json", total_imgs, category_names)
    reducer.reduce_data(output_file)
