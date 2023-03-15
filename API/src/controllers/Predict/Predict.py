from flask import Flask, render_template, request, send_file

from flask import Markup
import os
from src import app
import random
from io import BytesIO
import cv2
import torch
from numpy import random
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

def check_image(images_name):
    list_img = ['png', 'PNG', 'jpg', 'JPG', 'TIF']
    for i in list_img:
        x = images_name.endswith(i)
        if x is True:
            return x


weights = 'weights/best.pt'
set_logging()
device = select_device('')
half = device.type != 'cpu'
imgsz = 512
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
if half:
    model.half()  # to FP16


@app.route('/yolov5x', methods=['POST'])
def predict():
    if request.files.get("image"):
        image = request.files["image"]
        print(image.filename)
        if check_image(image.filename):
            source = os.path.join("images", image.filename)  # source = 'images/019465.TIF'

            # Read Image
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # Get names and colors
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

            conf_thres = 0.5
            iou_thres = 0.5

            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=False)[0]
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

            extra = ""
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                save_path = os.path.join("results", image.filename)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        extra += "<br>- <b>" + str(names[int(cls)]) + "</b> with conf <b>{:.2f} </b>".format(conf)
                result_image = Image.fromarray(im0)
            return serve_pil_image(result_image)
        else:
            return 'The file you sent is not an image, please check the file again!'

    else:
        return 'Ask to select file: png, jpg, TIF, ...'





