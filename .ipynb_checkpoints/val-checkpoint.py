import torch
import pdb
from ultralytics import YOLO

# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

# model_t = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train32/weights/best.pt')
# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/coco_v8l/weights/best.pt')
# model_t = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

# model_t = YOLO('/hy-tmp/yolov8_Distillation/runs/detect/train7/weights/best.pt')
# success = modelL.export(format="onnx",device="cpu")

data = "ultralytics/datasets/coco.yaml"
# model_t.model.model[-1].set_Distillation = True

# model_t.train(data=data, epochs=100, imgsz=640, Distillation = None)

# model_s = YOLO('/hy-tmp/yolov8_Distillation/runs/detect/train6/weights/best.pt')
model_s = YOLO('/hy-tmp/yolov8_Distillation/runs/detect/yolov5+belt/weights/best.pt')


# success = modeln.export(format="onnx")
# modelL.val(data=data)

# model_s.train(data=data, epochs=100, imgsz=640, Distillation = model_t.model)
model_s.val(data=data, batch=1, imgsz=640, Distillation = None)



