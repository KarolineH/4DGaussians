# Imports from D3DG project
import torch
import numpy as np
import open3d as o3d
import time
import os
import sys

# Make it possibel to find the Dynamic3DGaussians package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'Dynamic3DGaussians'))
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy

# Imports from 4DGS project
from scene import Scene
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from utils.general_utils import safe_state
from gaussian_renderer import render
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

w, h = 500, 800
near, far = 0.01, 100.0
view_scale = 3
fps = 1
traj_frac = 10  # 4% of points
traj_length = 10
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()


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

def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k

def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets

def calculate_trajectories(coords):
    # coords is an np array of shape (num_frames, num_points, 3)
    # first keep only the specified fraction of points
    pts = coords[:,:int(np.round(coords.shape[1]/traj_frac)),:]
    num_lines = len(pts[0])
    cols = np.repeat(colormap[np.arange(len(pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(pts))[traj_length:]:
        out_pts.append(np.array(pts[t - traj_length:t + 1]).reshape(-1, 3))
    return make_lineset(out_pts, cols, num_lines)

def visualize(seq, exp):
    # Fetch scene data 
    scene, gaussians, pipeline, background, cam_type = get_scene(exp, seq)

    # Initialise viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)
    w2c, k = init_camera() # Iinitial camera position
    if not os.path.exists(f'/workspace/4DGaussians/output/{exp}/{seq}/trajectory'):
        os.makedirs(f'/workspace/4DGaussians/output/{exp}/{seq}/trajectory')

    # Get all the point clouds across all frames
    point_coordinates = []
    for frame in range(scene.maxtime):
        view = define_view(w,h,k,w2c,near,far,frame_nr=frame,max_time=scene.maxtime)
        render_output, means3D = render(view, gaussians, pipeline, background,cam_type=cam_type, return_extra_info=True)
        point_coordinates.append(means3D.detach().double().cpu().numpy())
    point_coordinates = np.array(point_coordinates)

    # compute the trajectories for a subset of the points
    linesets = calculate_trajectories(point_coordinates)

    # View control setup
    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c # This sets the camera position
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)
    render_options = vis.get_render_option()
    render_options.point_size = view_scale*0.75
    render_options.light_on = False
    # render_options.point_size=3.0
    z_range = [point_coordinates[:,:,2].max(), point_coordinates[:,:,2].min()]

    # Add initial point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_coordinates[traj_length,:,:])
    vis.add_geometry(pcd)
    # Add intial lineset
    lines = o3d.geometry.LineSet()
    lines.points = linesets[0].points
    lines.colors = linesets[0].colors
    lines.lines = linesets[0].lines
    vis.add_geometry(lines)

    for frame in range(scene.maxtime-traj_length):
        # starting from the first frame with a 'full' trajectory history, update point cloud and lines 
        pcd.points = o3d.utility.Vector3dVector(point_coordinates[frame+traj_length,:,:])
        # colour the points based on their z-coordinate in gray scale
        zs = point_coordinates[frame+traj_length,:,2]
        normalised_zs = 0.8*(zs - z_range[1]) / (z_range[0] - z_range[1])
        colors = np.stack((1-normalised_zs, 1-normalised_zs, 1-normalised_zs), axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(pcd)
        lines.points = linesets[frame].points
        lines.colors = linesets[frame].colors
        lines.lines = linesets[frame].lines
        vis.update_geometry(lines)

        if not vis.poll_events():
            break
        vis.update_renderer()
        vis.capture_screen_image(f'/workspace/4DGaussians/output/{exp}/{seq}/trajectory/frame_{frame:04d}.png', do_render=True)
    
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":

    exp_name = "test"
    for sequence in ['rotation', 'shedding']:# ["ani_growth", "bending", "branching", "colour", "hole", "rotation", "shedding", "stretching", "translation", "twisting", "uni_growth"]:
        visualize(sequence, exp_name)
