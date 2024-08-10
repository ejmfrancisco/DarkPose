import os
import sys
import torch
import onnx
from lib.models import pose_resnet
from lib.core.config import config as cfg
from lib.core.config import update_config
sys.path.append('./lib')

pose_config_path     = './experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3_modify.yaml'
pose_model_path      = './output/coco/pose_resnet_50/256x192_d256x3_adam_lr1e-3_modify/final_state.pth.tar'
pose_model_onnx_path = './'
dynamic = False 
half = True 
batch = 20 


if __name__ == '__main__':
    ctx = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load model
    update_config(pose_config_path)
    pose_model = pose_resnet.get_pose_net(cfg, is_train=False)
    pose_model.load_state_dict(torch.load(pose_model_path))
    pose_model.to(ctx)
    if half == True:
        pose_model.half()
    pose_model.eval()

    #dummy input
    dummy_input = torch.randn((batch,3,256,192)).to(ctx)
    if half == True:
        dummy_input = dummy_input.half()

    # config onnx
    floating_point_type = '_FP32'
    if half == True:
        floating_point_type = '_FP16'

    pose_onnx_name = 'pose_batch_'+str(batch)+floating_point_type+'.onnx'
    dynamic_axes = None
    if dynamic == True:
        pose_onnx_name = 'pose_batch_'+'dynamic'+floating_point_type+'.onnx'
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    pose_onnx_name = os.path.join(pose_model_onnx_path, pose_onnx_name)

    # export to onnx
    torch.onnx.export(pose_model, dummy_input, pose_onnx_name, 
                      input_names=['input'], output_names=['output'],
                      export_params=True, dynamic_axes=dynamic_axes)

    # validate onnx model
    model_onnx = onnx.load(pose_onnx_name)
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, pose_onnx_name)


