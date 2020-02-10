

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


for path in `ls models/*modelE*`
do
    echo $path
#python evaluate.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --lmbda 0 --threshold 0.5 --path models/"$path"
    python evaluate.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --lmbda 0 --threshold 0.5 --path "$path"

done


