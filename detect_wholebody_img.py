import argparse
import os
import shutil
import time
import numpy as np 
import sys

import cv2
import torch
import torchvision
import torchvision.transforms as transforms

# from yolov5.utils.general import non_max_suppression, scale_coords
# from yolov5.utils.datasets import letterbox

from yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh,plot_one_box)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from lib.models import pose_resnet
from lib.core.inference import get_final_preds

# from config import cfg
# from config import update_config
from lib.core.config import update_config 
from lib.core.config import config as cfg

from lib.utils.transforms import get_affine_transform
from tools import box_to_center_scale

currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

def draw_poses(drawingImg, keypts, vis_thresh=0.2):
    """
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
        17: "left_toe",
        20: "right_toe",
        19: "left_heel",
        23: "right_heel",
        100: "left_hand",
        121: "right_hand"
    }
    """
    keyPair_body = [(0,1), (0,2), (1,2), (7,8), (1,7), (2,8)]
    keyPair_left = [(1,3), (3,5), (5,11), (7,9)]
    keyPair_right = [(2,4), (4,6), (6,12), (8,10)]

    # keyPair_left = [(5,7), (7,9), (11,13), (13,15)]
    # keyPair_right = [(6,8), (8,10), (12,14),(14,16)]

    poolypts_body = []
    poolypts_body_score = []
    for i in range(len(keyPair_body)):
        idx1 = keyPair_body[i][0]
        idx2 = keyPair_body[i][1]
        pt1 = keypts[idx1][0:2]
        pt2 = keypts[idx2][0:2]
        pair = (pt1,pt2)
        score = min(keypts[idx1][2], keypts[idx2][2])
        poolypts_body.append(pair)
        poolypts_body_score.append(score)

    poolypts_left = []
    poolypts_left_score = []
    for i in range(len(keyPair_left)):
        idx1 = keyPair_left[i][0]
        idx2 = keyPair_left[i][1]        
        pt1 = keypts[idx1][0:2]
        pt2 = keypts[idx2][0:2]
        pair = (pt1,pt2)
        score = min(keypts[idx1][2], keypts[idx2][2])
        poolypts_left.append(pair)
        poolypts_left_score.append(score)

    poolypts_right = []
    poolypts_right_score = []
    for i in range(len(keyPair_right)):
        idx1 = keyPair_right[i][0]
        idx2 = keyPair_right[i][1]
        pt1 = keypts[idx1][0:2]
        pt2 = keypts[idx2][0:2]
        pair = (pt1,pt2)
        score = min(keypts[idx1][2], keypts[idx2][2])
        poolypts_right.append(pair)
        poolypts_right_score.append(score)
 
    body_polypts = np.int32(poolypts_body)
    poolypts_left = np.int32(poolypts_left)
    poolypts_right = np.int32(poolypts_right)

    # draw keypoints
    for k, keypt in enumerate(keypts):
        if keypt[2] > vis_thresh:
            cv2.circle(drawingImg, (int(keypt[0]), int(keypt[1])), 3, (255, 0, 0), 1)
    
    # Draw edges
    for i in range(len(body_polypts)):
        pt1 = (int(poolypts_body[i][0][0]), int(poolypts_body[i][0][1]))
        pt2 = (int(poolypts_body[i][1][0]), int(poolypts_body[i][1][1]))
        if poolypts_body_score[i] > vis_thresh:
            drawingImg = cv2.line(drawingImg, pt1, pt2, color=(0,255,255),thickness=1)

    for i in range(len(poolypts_left)):
        pt1 = (int(poolypts_left[i][0][0]), int(poolypts_left[i][0][1]))
        pt2 = (int(poolypts_left[i][1][0]), int(poolypts_left[i][1][1]))
        if poolypts_left_score[i] > vis_thresh:
            drawingImg = cv2.line(drawingImg, pt1, pt2, color=(0,0,255),thickness=1)

    for i in range(len(poolypts_right)):
        pt1 = (int(poolypts_right[i][0][0]), int(poolypts_right[i][0][1]))
        pt2 = (int(poolypts_right[i][1][0]), int(poolypts_right[i][1][1]))
        if poolypts_right_score[i] > vis_thresh:
            drawingImg = cv2.line(drawingImg, pt1, pt2, color=(0,255,0),thickness=1)




if __name__ == '__main__':
    img_path        = '20221130_014735_906.jpg'
    video_file      = '/home/tricubics/eric/inference/lab/4p-c0.avi'
    yolo_hub = '/home/tricubics/eric/SimpleBase_DARK_inference/yolov5/ultralytics_yolov5_master'
    detector_weight = '/home/tricubics/eric/SimpleBase_DARK_inference/yolov5/yolov5l.pt'
    pose_weight     = '/media/tricubics/big/thienpm/openstore_perception_thienpm/data/checkpoints/pose/pose_13kpt_final.pth.tar'
    # pose_cfg_path   = '/media/tricubics/big/thinh/wholebody_pose/YOLOv5_ShelfNet/shelfnet/res50_256x192_d256x3_adam_lr1e-3_wholebody.yaml'
    pose_cfg_path   = '/media/tricubics/big/thienpm/openstore_perception_thienpm/data/checkpoints/pose/256x192_d256x3_adam_lr1e-3_modify.yaml'
    img_size = 640
    conf_thres = 0.7
    iou_thres = 0.5
    classes = 0 # human person
    ctx = torch.device('cuda') 
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cfg.merge_from_file(pose_cfg_path)
    update_config(pose_cfg_path)
    pose_model = pose_resnet.get_pose_net(cfg)
    pose_model.load_state_dict(torch.load(pose_weight))
    pose_model.cuda()
    pose_model.eval()

    sys.path.insert(0, yolo_hub)
    # detector = torch.load(detector_weight)['model'].float()  # load to FP32
    detector = torch.load(detector_weight, map_location=ctx)['model'].float()
    detector.cuda().eval()


    # img_src  = cv2.imread(img_path)

    # Loading an video
    vidcap = cv2.VideoCapture(video_file)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    count = 0

    # List of image file names
    path = '/home/tricubics/eric/image/test/192.168.0.111'
    images = [f for f in os.listdir(path) if f.endswith('.jpg')]
    images = sorted(images)
    for img_file in images:
        total_now = time.time()
        img_path = os.path.join(path,img_file)
        img_src = cv2.imread(img_path)

        # Check if the image was successfully loaded
        if img_src is None:
            print(f'Failed to load image: {img_path}')
            continue

        # Padded resize
        image_whole = letterbox(img_src,new_shape=img_size)[0]
        # Convert
        image_whole = image_whole[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image_whole = np.ascontiguousarray(image_whole)
        # numpy to tensor
        image_whole = torch.from_numpy(image_whole).cuda()
        image_whole = image_whole/255.0  # 0 - 255 to 0.0 - 1.0
        if image_whole.ndimension() == 3:
            image_whole = image_whole.unsqueeze(0)

        # Detection ********
        # Inference
        with torch.no_grad():
            pred = detector(image_whole, augment=False)[0]  # list: bz * [ (#obj, 6)]
        
        # Apply NMS and filter object other than person (cls:0)
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                    classes=classes, agnostic=False)
        # get all obj 
        det = pred[0]  # for video, bz is 1 

        if det is not None:
            if det.shape[0] == 1:
                if len((det[:,5:6] == 0).T.squeeze().nonzero().squeeze()) != 0:
                    det = torch.index_select(det,0, (det[:,5:6] == 0).T.squeeze().nonzero().squeeze())
            else:
                det = torch.index_select(det,0, (det[:,5:6] == 0).T.squeeze().nonzero().squeeze())

        # find human only in the frame
        if det is not None and len(det)>=1:  # det: (#obj, 6)  x1 y1 x2 y2 conf cls
            # Rescale boxes from img_size to original im0 size
            det[:, :4] = scale_coords(image_whole.shape[2:], det[:, :4], img_src.shape).round()
            pred_boxes = det[:,:4].cpu().numpy().reshape(-1,2,2)

            # pose estimation : for multiple people
            model_inputs = []
            centers = []
            scales = []
            for cnt_box in range(pred_boxes.shape[0]):
                center, scale = box_to_center_scale(pred_boxes[cnt_box], 
                                                    model_image_width= img_src.shape[1],
                                                    model_image_height= img_src.shape[0], pixel_std=200)
                trans = get_affine_transform(center, scale, 0, cfg.MODEL.IMAGE_SIZE)
                model_img_w, model_img_h = int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])
                print(model_img_w, model_img_h)
                model_input = cv2.warpAffine(img_src, trans,(model_img_w, model_img_h), flags=cv2.INTER_LINEAR) # rgb
                print('=====> model_inputs: ', model_input)
                model_input = pose_transform(model_input)#.unsqueeze(0)
                model_inputs.append(model_input)
                centers.append(center)
                scales.append(scale)

                x1 = int(pred_boxes[cnt_box][0][0])
                y1 = int(pred_boxes[cnt_box][0][1])
                x2 = int(pred_boxes[cnt_box][1][0])
                y2 = int(pred_boxes[cnt_box][1][1])

                cv2.rectangle(img_src, (x1, y1), (x2, y2), (255,0,0), 2)

            # n * 1chw -> nchw
            model_inputs = torch.stack(model_inputs)

            # compute output heatmap
            output = pose_model(model_inputs.cuda())
            coords, maxvals = get_final_preds(
                                              cfg,
                                              output.cpu().detach().numpy(),
                                              np.asarray(centers),
                                              np.asarray(scales))
            pose_comb = np.concatenate((coords,maxvals),2)

            for pose_c in pose_comb:
                draw_poses(img_src, pose_c, 0.2)

            image_show = cv2.resize(img_src,(1024,768),cv2.INTER_CUBIC)
            cv2.imshow('frame.png', image_show)
            k = cv2.waitKey(0)
            if k == 27:  # Esc button
                vidcap.stop()
                break


