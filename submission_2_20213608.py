import os
import cv2
import yaml
import torch
import random
import numpy as np
from PIL import Image
from datetime import datetime
from models import YOLOv9t
from ultralytics import YOLO

def submission_2_20213608(yaml_path, output_json_path):
    data_config = load_yaml_config(yaml_path)
    model_name = 'yolov9t'
    ex_dict = {}
    epochs = 20
    batch_size = 8
    optimizer = 'SGD'
    lr = 0.01
    momentum = 0.937
    weight_decay = 5e-4

    Experiments_Time = datetime.now().strftime("%y%m%d_%H%M%S")
    ex_dict['Iteration']  = int(yaml_path.split('.yaml')[0][-2:])
    image_size = 640
    output_dir ='tmp'
    devices = [0]
    device = torch.device("cuda:"+str(devices[0])) if torch.cuda.is_available() else torch.device("cpu")

    ex_dict['Experiment Time'] = Experiments_Time
    ex_dict['Epochs'] = epochs
    ex_dict['Batch Size'] = batch_size
    ex_dict['Device'] = device
    ex_dict['Optimizer'] = optimizer
    ex_dict['LR'] = lr
    ex_dict['Weight Decay'] = weight_decay
    ex_dict['Momentum'] = momentum
    ex_dict['Image Size'] = image_size
    ex_dict['Output Dir'] = output_dir 
    Dataset_Name = yaml_path.split('/')[1]
    ex_dict['Dataset Name'] = Dataset_Name
    ex_dict['Data Config'] = yaml_path
    ex_dict['Number of Classes'] = data_config['nc']
    ex_dict['Class Names'] = data_config['names']

    control_random_seed(42)
    model = YOLO('models/YOLOv9t/yolov9t.yaml', verbose=False)
    os.makedirs(output_dir, exist_ok=True)
    ex_dict['Model Name'] = model_name
    ex_dict['Model'] = model

    ex_dict = YOLOv9t.train_model(ex_dict)
    test_images = get_test_images(data_config)
    results_dict = detect_and_save_bboxes(ex_dict['Model'], test_images)
    save_results_to_file(results_dict, output_json_path)

# 이하 공통 함수
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_test_images(config):
    test_path = config['test']
    root_path = config['path']
    test_path = os.path.join(root_path, test_path)
    
    if os.path.isdir(test_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    elif test_path.endswith('.txt'):
        with open(test_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        return image_paths

def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        torch.backends.cudnn.benchmark = False

def detect_and_save_bboxes(model, image_paths):
    results_dict = {}
    for img_path in image_paths:
        results = model(img_path, verbose=False, task='detect')
        img_results = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                img_results.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results
    return results_dict

def save_results_to_file(results_dict, output_path):
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"결과가 {output_path}에 저장되었습니다.")


