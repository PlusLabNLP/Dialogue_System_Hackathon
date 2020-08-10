CUDA_VISIBLE_DEVICES=3 python train.py data/ ./ ./ 2>&1 |tee trainlog-$(date +%y.%m.%d.-%H:%M:%S).txt
