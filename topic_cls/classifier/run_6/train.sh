CUDA_VISIBLE_DEVICES=3 python train.py data/ ./ ./ 2>&1 |tee logs/trainlog-$(date +%y.%m.%d.-%H:%M:%S).txt
