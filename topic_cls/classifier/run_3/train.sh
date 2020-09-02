TIME=$(date +%y.%m.%d-%H:%M:%S)
mkdir logs/train-$TIME
mkdir logs/train-$TIME/data
cp data/* logs/train-$TIME/data
CUDA_VISIBLE_DEVICES=$1 python train.py data/ ./ ./ 2>&1 |tee logs/train-$TIME/trainlog-$TIME.txt
#$(date +%y.%m.%d.-%H:%M:%S).txt

