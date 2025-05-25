import os
import sys
import numpy as np
import torch

# Imports from 4DGS project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene import Scene
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from utils.general_utils import safe_state
from gaussian_renderer import render
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

'''
Code for rendering out specific views of a 4D Gaussian scene.
Specify the scene (experiment name and sequence name), the camera parameters (w2c, k)
and the frame number.
'''


def get_combined_args(parser, exp_name, seq_name):
    # Find the trained model by its experiment name and sequence/scene name 
    cmdlne_string = ['--model_path', f'/workspace/4DGaussians/output/{exp_name}/{seq_name}/']
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    # Find the cfg_args file in the model path
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    # Extract the arguments from the config file
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def get_scene(exp_name, seq_name):
    parser = ArgumentParser(description="visualisation parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", default='/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py', type=str)
    args = get_combined_args(parser, exp_name, seq_name)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    dataset = model.extract(args)
    hyperparam = hyperparam.extract(args)
    iteration = args.iteration
    pipeline = pipeline.extract(args)
    with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree, hyperparam)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            cam_type=scene.dataset_type
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return scene, gaussians, pipeline, background, cam_type

def render_single(view, gaussians, pipeline, background, cam_type):
    render_output = render(view, gaussians, pipeline, background,cam_type=cam_type)
    image = render_output['render'] # colour image rendering
    depth = render_output['depth'] # depth image rendering
    return image, depth


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

def setup_camera(w,h,k,w2c,near,far):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam

def define_view(w,h,k,w2c,near,far,frame_nr=0,max_time=48):
    cam = setup_camera(w,h,k,w2c,near,far)
    time_frac = frame_nr / max_time # frame numbers start at 0 and go to max_time-1
    cam_info = {"camera":cam,"time":time_frac,"image":None}
    return cam_info

if __name__ == "__main__":

    w, h = 640, 360
    near, far = 0.01, 100.0
    show = True

    exp_name = "test"
    for sequence in ['rotation']:# ["ani_growth", "bending", "branching", "colour", "hole", "rotation", "shedding", "stretching", "translation", "twisting", "uni_growth"]:
        scene, gaussians, pipeline, background, cam_type = get_scene(exp_name, sequence)

        w2c, k = init_camera() #Example camera, specify your own camera parameters here
        frame_nr = 0

        view = define_view(w,h,k,w2c,near,far,frame_nr=frame_nr,max_time=scene.maxtime)
        image, depth = render_single(view, gaussians, pipeline, background, cam_type)

        if show:
            import matplotlib.pyplot as plt
            plt.imshow(image.permute(1,2,0).contiguous().double().detach().cpu().numpy())
            plt.show()