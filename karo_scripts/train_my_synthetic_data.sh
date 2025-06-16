exp_name1="mine_05"

python train.py -s /workspace/synthetic_data/ani_growth/ --port 6017 --expname "$exp_name1/ani_growth" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/bending/ --port 6017 --expname "$exp_name1/bending" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/branching/ --port 6017 --expname "$exp_name1/branching" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/colour/ --port 6017 --expname "$exp_name1/colour" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/hole/ --port 6017 --expname "$exp_name1/hole" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/rotation/ --port 6017 --expname "$exp_name1/rotation" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/shedding/ --port 6017 --expname "$exp_name1/shedding" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/stretching/ --port 6017 --expname "$exp_name1/stretching" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/translation/ --port 6017 --expname "$exp_name1/translation" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/twisting/ --port 6017 --expname "$exp_name1/twisting" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/uni_growth/ --port 6017 --expname "$exp_name1/uni_growth" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait
python train.py -s /workspace/synthetic_data/large_growth/ --port 6017 --expname "$exp_name1/large_growth" --configs arguments/my_synthetic_data/my_synthetic_data.py
wait

echo "Done"