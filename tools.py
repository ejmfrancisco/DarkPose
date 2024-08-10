import os 
import numpy as np
import ruamel.yaml


def save_yaml(dict_data, file_path):
    import ruamel.yaml
    yaml = ruamel.yaml.YAML()
    yaml.version = (1,2)
    yaml.default_flow_style = None

    with open(file_path, 'w') as outfile:
        yaml.dump(dict_data, outfile)


def read_yaml(f_path):
        yaml = ruamel.yaml.YAML()
        yaml.version = (1,2)
        yaml.default_flow_style = None
        
        if(os.path.exists(f_path)):
            with open(f_path, 'r') as f:
                try:
                    data = yaml.load(f)
                    return data
                except Exception as exc:
                    print(exc)
        return None, None


def validate_cam_setting(camera_settings, cfg):
    """
    ensure all cameras must have same resolution & fps
    """
    cfg_resolution = (cfg.CAMERA.WIDTH, cfg.CAMERA.HEIGHT, cfg.CAMERA.CHANNEL)
    cfg_fps = cfg.CAMERA.FPS

    cameras_resolution = [tuple(cam['resolution']) for cam in camera_settings.values()]
    cameras_fps = [cam['fps'] for cam in camera_settings.values()]

    for cam_res, cam_fps in zip(cameras_resolution, cameras_fps):
        if (cam_res != cfg_resolution) or (cam_fps != cfg_fps):
            return False
    return True


def load_cam_settings(f_path, cam_ips):
    # load camera configs from path
    camera_info = read_yaml(f_path)
    camera_setting = {ip:setting for ip, setting in camera_info.items() if ip in cam_ips}
    return camera_setting


def load_camip_infos(f_path):
    """
    loads video_ids from folder data/videos/video_id.yaml    
    """
    video_id_raw = read_yaml(f_path)
    return list(video_id_raw.keys())


def box_to_center_scale(box, model_image_width, model_image_height,pixel_std=200):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale