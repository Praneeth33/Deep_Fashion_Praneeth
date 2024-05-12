
# Deep_Fashion 
#### Matrice.ai coding assignment


## Problem:
1. To download the Deep_Fashion dataset from s3 
2. Create a sample dataset from the huge dataset
3. Convert the MSCOCO annotations into YOLO format
4. Train a yolov9 instance segmentation model (while taking parameters from a config file)
5. Calculate the performance metrics and analyse the model
6. Optimize model for ONNX and OpenVino format, perform predictions using using those models.
7. Measure the throughput and compare ONNX, OpenVINO and YOLO formats
8. Create a docker file with CUDA and OpenVINO support that can be run for trainig model on GPU and inference on Intel CPU
9. Document code without linting errors

## Solution Approach:

1. Downoladed the Deep_Fashion dataset from the provided link

2. Created a Sample dataset of 500 images:  
    -> The main dataset having 13 classes when broken down to 500 images would result in only 25 images per class, which would have given bad output after training. So,  
    -> The dataset of 500 images contained only two classes namely "trousers" and "short sleve top"  
    -> The dataset contained an MSCOCO json corresponding 1,91,000 images. To convert the the huge json file to a smaller version corresponding to 500 images  
    > RUN---> "coco_reduce.py" by providing the required parameters

3. To create a YOLO annotations from MSCOCO annotations:  
    > RUN ---> "coco2yolo.py" by providing the required parameters
    
4. After YOLO annotations are created  
> RUN ---> "make_dataset_yolo_format.py"  
To split images and labels into train val and test sets as per YOLO requirement  
->  After the dataset is ready, download the "yolov9c-seg.pt" weights and store them in the weights directory  
->  Update the config files from the config directory and   
> RUN ---> "train.py"

5. My model had the MAP Score of 0.85. MAP is (Mean Average Precision) implying it's a good model. All the metric values have been uploaded in the repo under training_metrics directory. To improve the model performance we consider hyperparameter tuning upto 100 iterations to get the best parameters (compute intesive). 

6. Model can be Optimized into OpenVINO format using "optimize_openvino.py". Couldn't run the model due to hardware related errors.

7. According to the study by ultralytics model optimised in OpenVINO is best for intelsystems, it improves inference time drastically hence it is better than ONNX and Pytorch. The bar graph is available here -- "https://docs.ultralytics.com/integrations/openvino/?h=openvin#intel-flex-gpu"  

8. I have no experience in Docker, hence couldn't create a required training file.

9. The code is well documented and linting error free.

The models weights are available here incase if there is problem in downloading -->  https://drive.google.com/drive/folders/1FG2IOaDW8fOf28P8lji_WC0yEJZgYGmP?usp=sharing