expname="mine_05"
iter="20000"

python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/ani_growth/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/bending/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/branching/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/colour/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/hole/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/rotation/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/shedding/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/stretching/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/translation/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/twisting/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/uni_growth/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/4DGaussians/output/${expname}/large_growth/" --skip_train --skip_video --iteration $iter --configs "/workspace/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait

echo "Done"