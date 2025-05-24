import os
import cv2

experiments_dir = "/workspace/4DGaussians/output/mine/"
output_template = "video_{}.mp4"
frame_rate = 24 # fps
h = 800
w = 800

for folder in os.listdir(experiments_dir):
    folder_path = os.path.join(experiments_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    test_render_dir = os.path.join(folder_path, "test/ours_20000/renders")
    if not os.path.exists(test_render_dir):
        continue

    images = sorted([f for f in os.listdir(test_render_dir) if f.endswith(".png")])


    # Create video writers
    writers = [
        cv2.VideoWriter(os.path.join(test_render_dir,output_template.format(i)), cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h))
        for i in range(4)
    ]

    # Write frames to appropriate video
    for i, img_name in enumerate(images):
        frame = cv2.imread(os.path.join(test_render_dir, img_name))
        writers[i % 4].write(frame)

    # Release writers
    for writer in writers:
        writer.release()
