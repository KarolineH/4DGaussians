import os
import shutil

''' 
Fetches the videos for test view 1 (camera id 9?) from the Dynamic3DGaussians experiment directory
and copies them to a temporary directory. 
'''

model_dir = '/workspace/data/4dgs' 
exp_name = 'mine_07'
tmp_dir = '/workspace/tmp'
exp_dir = os.path.join(model_dir, exp_name)
for sequence in os.listdir(exp_dir):
    sequence_dir = os.path.join(exp_dir, sequence)
    vid_file = os.path.join(sequence_dir, 'test/ours_20000', 'colour_videos', 'view_1.mp4')
    shutil.copyfile(vid_file, os.path.join(tmp_dir, f'{exp_name}_{sequence}_view_1.mp4'))