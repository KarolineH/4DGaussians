expname="mine_05"

python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/ani_growth/"
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/bending/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/branching/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/colour/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/hole/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/rotation/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/shedding/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/stretching/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/translation/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/twisting/" 
wait
python metrics.py --model_path "/workspace/4DGaussians/output/${expname}/uni_growth/" 
wait

echo "Done"