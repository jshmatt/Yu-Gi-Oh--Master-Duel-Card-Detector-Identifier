from ultralytics import YOLO
from pathlib import Path

work_dir = Path.cwd()

# download annotated dataset from roboflow

# Load Yolo nano pretrained on COCO
model = YOLO(work_dir + 'YGOmodels/yolo26n.pt') 

# train the model
model.train(
    data=work_dir + '/YGO-Card-Classification/data.yaml', # change path to downloaded data.yaml
    epochs=400,
    imgsz=640,
    batch=16,
    name='yolo26_custom',
    device=0
)