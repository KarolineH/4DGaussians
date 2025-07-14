exp_name="mine_07"

python train.py -s /workspace/data/synthetic_data/ani_growth/ --port 6017 --expname "$exp_name/ani_growth" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/ani_growth"
wait
python train.py -s /workspace/data/synthetic_data/bending/ --port 6017 --expname "$exp_name/bending" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/bending"
wait
python train.py -s /workspace/data/synthetic_data/branching/ --port 6017 --expname "$exp_name/branching" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/branching"
wait
python train.py -s /workspace/data/synthetic_data/colour/ --port 6017 --expname "$exp_name/colour" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/colour"
wait
python train.py -s /workspace/data/synthetic_data/hole/ --port 6017 --expname "$exp_name/hole" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/hole"
wait
python train.py -s /workspace/data/synthetic_data/rotation/ --port 6017 --expname "$exp_name/rotation" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/rotation"
wait
python train.py -s /workspace/data/synthetic_data/shedding/ --port 6017 --expname "$exp_name/shedding" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/shedding"
wait
python train.py -s /workspace/data/synthetic_data/stretching/ --port 6017 --expname "$exp_name/stretching" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/stretching"
wait
python train.py -s /workspace/data/synthetic_data/translation/ --port 6017 --expname "$exp_name/translation" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/translation"
wait
python train.py -s /workspace/data/synthetic_data/twisting/ --port 6017 --expname "$exp_name/twisting" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/twisting"
wait
python train.py -s /workspace/data/synthetic_data/uni_growth/ --port 6017 --expname "$exp_name/uni_growth" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/uni_growth"
wait
python train.py -s /workspace/data/synthetic_data/large_growth/ --port 6017 --expname "$exp_name/large_growth" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/large_growth"
wait
python train.py -s /workspace/data/synthetic_data/uni_shrink/ --port 6017 --expname "$exp_name1/uni_shrink" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/uni_shrink"
wait
python train.py -s /workspace/data/synthetic_data/large_shrink/ --port 6017 --expname "$exp_name1/large_shrink" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/large_shrink"
wait
python train.py -s /workspace/data/synthetic_data/emergence/ --port 6017 --expname "$exp_name1/emergence" --configs arguments/my_synthetic_data/my_synthetic_data.py --model_path "/workspace/data/4dgs/$exp_name/emergence"
wait


iter="20000"

python render_full.py --model_path "/workspace/data/4dgs/$exp_name/ani_growth" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/bending/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/branching/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/colour/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/hole/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/rotation/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/shedding/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/stretching/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/translation/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/twisting/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/uni_growth/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/large_growth/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/uni_shrink/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/large_shrink/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait
python render_full.py --model_path "/workspace/data/4dgs/${exp_name}/emergence/" --skip_train --skip_video --iteration $iter --configs "/workspace/model/4DGaussians/arguments/my_synthetic_data/my_synthetic_data.py"
wait

echo "Rendering test results completed"

python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/ani_growth/"
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/bending/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/branching/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/colour/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/hole/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/rotation/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/shedding/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/stretching/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/translation/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/twisting/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/uni_growth/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/large_growth/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/uni_shrink/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/large_shrink/" 
wait
python metrics.py --model_path "/workspace/data/4dgs/${exp_name}/emergence/" 

echo "Metrics computed"