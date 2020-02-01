# 2 Feb | CIFAR100: PCN
python main_cifar.py --batch_size 128 --circles 0 >> cir0.log # T=0 PCN
python main_cifar.py --batch_size 128 --circles 1 >> cir1.log # T=1 PCN
python main_cifar.py --batch_size 128 --circles 2 >> cir2.log # T=1 PCN
# CIFAR100: vanilla
python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300 --step_all 0 --step_clf 0 --vanilla 1 >> vanilla.log # vanilla
# CIFAR100: vanilla + all35clf15
python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300 --step_all 35 --step_clf 15 --vanilla 1 >> vanilla_all35clf15_cifar100.log # vanilla
# CIFAR10: vanilla + all35clf15
python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar10 --max_epoch 300 --step_all 35 --step_clf 15 --vanilla 1 >> vanilla_all35clf15_cifar10.log # vanilla


# 1 Feb CIFAR100 + GE
#python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300 --step_all 0 --step_clf 0 --ge 1 >> cir0_ge.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir1_adaptive_ge.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 0 --max_epoch 300 --step_all 0 --step_clf 0 --ge 1 >> cir1_ge.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir1_adaptive_dropout0.5_ge.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 35 --step_clf 15 --ge 1 >> cir1_adaptive_dropout0.5_all35clf15_ge.log





# 31 Jan: CIFAR100
#python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300 --step_all 0 --step_clf 0 >> cir0.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 >> cir1_adaptive.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 0 --max_epoch 300 --step_all 0 --step_clf 0 >> cir1.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 >> cir1_adaptive_dropout0.5.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 35 --step_clf 15 >> cir1_adaptive_dropout0.5_all35clf15.log


#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar10 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 15 --step_clf 10


# python  main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar10 --max_epoch 300 --step_all 0 --step_clf 0 --vanilla 1 # vanilla
# python main_cifar.py --batch_size 128 --circles 0 # T=0 PCN
# python main_cifar.py --batch_size 128 --circles 1 # T=1 PCN
