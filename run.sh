python main.py --batch_size 128 --circles 2 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 100 >> 1.log
python main.py --batch_size 128 --circles 3 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 100 >> 2.log
python main.py --batch_size 128 --circles 4 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 100 >> 3.log
python main.py --batch_size 128 --circles 5 --backend modelA --dataset_name cifar100 --max_epoch 100 >> 4.log
python main.py --batch_size 128 --circles 5 --backend modelB --dataset_name cifar100 --max_epoch 100 >> 5.log
python main.py --batch_size 128 --circles 5 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 100 >> 6.log

# python main_cifar.py
