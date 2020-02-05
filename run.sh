# 5 Feb | prednetE
python main.py --batch_size 128 --circles 1 --backend modelE --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0.01 >> cir1_dropout0.5_ge_lmbda0.01.log
#python main_cifar.py --batch_size 128 --circles 0 --backend prednetE >> cir0.log



# 4 Feb | lmbda 0.01 0.1
#python main.py --batch_size 128 --circles 10 --backend modelE --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 0 --lmbda 0.01 2>/dev/null
#python main.py --batch_size 128 --circles 5 --backend modelE --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0.1 2>/dev/null
#python main.py --batch_size 128 --circles 1 --backend modelE --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0.01 2>/dev/null
#python main.py --batch_size 128 --circles 5 --backend modelE --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0.01 2>/dev/null

# 3 Feb | cifar100 modelD
#python main.py --batch_size 128 --circles 0 --backend modelE --dataset_name cifar100 --max_epoch 300 --step_all 0 --step_clf 0 --ge 1 --vanilla 1 >> vanilla.log
#python main.py --batch_size 128 --circles 1 --backend modelE --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir1_adaptive_dropout0.5_ge.log
#python main.py --batch_size 128 --circles 2 --backend modelE --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dropout0.5_ge.log
#python main.py --batch_size 128 --circles 0 --backend modelE --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir0_adaptive_dropout0.5_ge.log

# 3 Feb | CIFAR100 modelD
#python main.py --batch_size 128 --circles 1 --backend modelD --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir1_adaptive_dropout0.5_ge.log





# 2 Feb | switch
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --fb 0:0:1 >> cir1_adaptive_dropout0.5_ge_001.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --fb 0:1:0 >> cir1_adaptive_dropout0.5_ge_010.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --fb 0:1:1 >> cir1_adaptive_dropout0.5_ge_011.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --fb 1:0:0 >> cir1_adaptive_dropout0.5_ge_100.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --fb 1:0:1 >> cir1_adaptive_dropout0.5_ge_101.log
#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --fb 1:1:0 >> cir1_adaptive_dropout0.5_ge_110.log

# 2 Feb | CIFAR100: PCN
#python main_cifar.py --batch_size 128 --circles 0 >> cir0.log # T=0 PCN
#python main_cifar.py --batch_size 128 --circles 1 >> cir1.log # T=1 PCN
#python main_cifar.py --batch_size 128 --circles 2 >> cir2.log # T=1 PCN
# CIFAR100: vanilla
#python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300 --step_all 0 --step_clf 0 --vanilla 1 >> vanilla.log # vanilla
# CIFAR100: vanilla + all35clf15
#python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300 --step_all 35 --step_clf 15 --vanilla 1 >> vanilla_all35clf15_cifar100.log # vanilla
# CIFAR10: vanilla + all35clf15
#python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar10 --max_epoch 300 --step_all 35 --step_clf 15 --vanilla 1 >> vanilla_all35clf15_cifar10.log # vanilla


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
