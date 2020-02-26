python evaluate.py --batch_size 128 --circles 5 --backend modelC_dp2 --dataset_name cifar100 --lmbda 0 --mode random --path models/modelC_dp2_cifar100_adaptive0_circles5_dropout1.00_all0clf0_vanilla0_ge1_fb111_lmbda0.0000.pt > debug.log
#for threshold in range 0.5 0.6 0.7 0.8 0.9 0.95
#do
#    echo "$threshold"
#    python evaluate.py --batch_size 128 --circles 5 --backend modelC_dp2 --dataset_name cifar100 --lmbda 0 --mode threshold --threshold "$threshold" --path models/modelC_dp2_cifar100_adaptive0_circles5_dropout1.00_all0clf0_vanilla0_ge1_fb111_lmbda0.0000.pt >> debug.log
#done
#python evaluate.py --batch_size 128 --circles 5 --backend modelC_dp2 --dataset_name cifar100 --lmbda 0 --mode random --path models/modelC_dp2_cifar100_adaptive0_circles5_dropout1.00_all0clf0_vanilla0_ge1_fb111_lmbda0.0000.pt > debug.log

#path='models/modelE_cifar100_adaptive0_circles1_dropout1.00_all0clf0_vanilla0_ge1_fb111.pt'
#echo "$path"
#python evaluate.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --lmbda 0 --threshold 0.5 --path "$path"
#
#path='models/modelE_cifar100_adaptive0_circles2_dropout1.00_all0clf0_vanilla0_ge1_fb111.pt'
#echo "$path"
#python evaluate.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --lmbda 0 --threshold 0.5 --path "$path"
#
#path='models/modelE_cifar100_adaptive0_circles5_dropout1.00_all0clf0_vanilla0_ge1_fb111.pt'
#echo "$path"
#python evaluate.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --lmbda 0 --threshold 0.5 --path "$path"


#for path in `ls models/*modelE*`
#do
#    echo $path
##python evaluate.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --lmbda 0 --threshold 0.5 --path models/"$path"
#    python evaluate.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --lmbda 0 --threshold 0.5 --path "$path"
#
#done


