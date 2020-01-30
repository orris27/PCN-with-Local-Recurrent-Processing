# 31 Jan: CIFAR100
python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300 --avg 0 --step_all 0 --step_clf 0 >> cir0.log
python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 1.0 --avg 0 --step_all 0 --step_clf 0 >> cir1_adaptive.log
python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 0 --max_epoch 300 --avg 0 --step_all 0 --step_clf 0 >> cir1.log
python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --avg 0 --step_all 0 --step_clf 0 >> cir1_adaptive_dropout0.5.log
python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --avg 0 --step_all 35 --step_clf 15 >> cir1_adaptive_dropout0.5_all35clf15.log


#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar10 --adaptive 1 --max_epoch 300 --dropout 0.5 --avg 0 --step_all 15 --step_clf 10


# python  main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar10 --max_epoch 300 --step_all 0 --step_clf 0 --vanilla 1 # vanilla
# python main_cifar.py --batch_size 128 --circles 0 # T=0 PCN
# python main_cifar.py --batch_size 128 --circles 1 # T=1 PCN
