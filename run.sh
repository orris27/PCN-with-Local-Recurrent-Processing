python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar10 --adaptive 1 --max_epoch 300 --dropout 0.5 --avg 1  >> cir1_adaptive_dropout0.5_avg.log
python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar10 --adaptive 0 --max_epoch 300 --dropout 0.5 --avg 1  >> cir1_dropout0.5_avg.log
python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar10 --max_epoch 300 --dropout 0.5 --avg 1  >> cir0_dropout0.5_avg.log

# python main_cifar.py
