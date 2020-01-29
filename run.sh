python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar10 --adaptive 1 --max_epoch 300  >> cir1_adaptive.log
python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar10 --adaptive 0 --max_epoch 300  >> cir1.log
python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar10 --max_epoch 300  >> cir0.log

python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300  >> cir1_adaptive.log
python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 0 --max_epoch 300  >> cir1.log
python main.py --batch_size 128 --circles 0 --backend modelC --dataset_name cifar100 --max_epoch 300  >> cir0.log

# python main_cifar.py
