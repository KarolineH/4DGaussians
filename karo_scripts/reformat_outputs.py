import os
import cv2


def test_renders_to_mp4(exp_dir, out_dir_name=None):
    '''
    Goes over all trained models in the experiment directory and converts the renders
    to mp4 videos.
    Assumes 4 test views per model and that images are interleaved.
    '''

    if out_dir_name is None:
        out_dir_name = 'colour_videos'

    output_template = "view_{}.mp4"
    frame_rate = 24 # fps
    h = 800
    w = 800

    for sequence in os.listdir(exp_dir):
        folder_path = os.path.join(exp_dir, sequence)
        if not os.path.isdir(folder_path):
            continue
        models = os.listdir(os.path.join(folder_path, "test"))
        for model in models:
            render_dir = os.path.join(folder_path, "test", model, "renders")
            if not os.path.exists(render_dir):
                continue

            if not os.path.isdir(os.path.join(folder_path, "test", model,out_dir_name)):
                os.makedirs(os.path.join(folder_path, "test", model, out_dir_name))

            images = sorted([f for f in os.listdir(render_dir) if f.endswith(".png")])

            # Create video writers
            # One for each of the four test views, because images are interleaved
            writers = [
                cv2.VideoWriter(os.path.join(folder_path, "test", model,out_dir_name,output_template.format(i)), cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))
                for i in range(4)
            ]

            # Write frames to appropriate video
            for j, img_name in enumerate(images):
                frame = cv2.imread(os.path.join(render_dir, img_name))
                writers[j % 4].write(frame)

            # Release writers
            # import subprocess
            for i,writer in enumerate(writers):
                writer.release()
                fix_video_codec(os.path.join(folder_path, "test", model,out_dir_name,output_template.format(i)))


def fix_video_codec(outpath):
    import subprocess
    temp_path = outpath + '.tmp.mp4'
    cmd = [
        'ffmpeg', '-i', outpath,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-movflags', '+faststart',
        temp_path,
        '-y'
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(temp_path, outpath)  # atomically overwrite original
        print(f"✔ Converted: {outpath}")
    except subprocess.CalledProcessError as e:
        print(f"❌ ffmpeg failed on {outpath}:", e)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return

if __name__ == "__main__":
    exp_dir = "/workspace/data/4dgs/mine_07/"
    test_renders_to_mp4(exp_dir)
    
