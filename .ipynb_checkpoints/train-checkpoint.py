import torch
import pdb
from ultralytics import YOLO

# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

model_t = YOLO('/hy-tmp/yolov8_Distillation/runs/detect/yolov8s/weights/best.pt')
# modelL = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/coco_v8l/weights/best.pt')
# model_t = YOLO('/home/huangzm/code/mycode/pytorch/ultralytics/runs/detect/train37/weights/best.pt')

# model_t = YOLO('/hy-tmp/yolov8_Distillation/runs/detect/yolov8s/weights/best.pt')
# success = modelL.export(format="onnx",device="cpu")

data = "ultralytics/datasets/coco.yaml"
# model_t.model.model[-1].set_Distillation = True

# model_t.train(data=data, epochs=100, imgsz=640, Distillation = None)

model_s = YOLO('/hy-tmp/yolov8_Distillation/runs/detect/yolov8nbelt/weights/best.pt')
# model_s = YOLO('yolov8s.pt')
# model_s = YOLO('yolov8l.pt')


# success = modeln.export(format="onnx")
# modelL.val(data=data)

model_s.train(data=data, epochs=100, imgsz=640, Distillation = model_t.model)
# model_s.train(data=data, epochs=100, imgsz=640, Distillation = None)



