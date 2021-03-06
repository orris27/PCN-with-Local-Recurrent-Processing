# 24 Feb
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 3.0 --gamma 0.8  > results/cifar100/modelG/2con3/cir5_ge_kd_T3.0_lmbda0.8.log
python main.py --batch_size 128 --circles 5 --backend resnet56_2con3_conv --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --attention no --T 2.0 --gamma 0.5 > results/cifar100/resnet/2con3/cir5_ge_2con3_conv_T2.0_gamma0.5.log
#python main.py --batch_size 128 --circles 0 --backend resnet56_2con3_conv --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --attention no --T 2.0 --gamma 0.5 > results/cifar100/resnet/2con3/cir0_ge_2con3_conv_T2.0_gamma0.5.log
#python main.py --batch_size 128 --circles 0 --backend resnet56_2con3_conv --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention no > results/cifar100/resnet/2con3/cir0_ge_2con3_conv.log

# 23 Feb
#python main.py --batch_size 128 --circles 0 --backend resnet56_2con3_att --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention scan > results/cifar100/resnet/2con3/cir0_ge_2con3_scan.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_2con3_att --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention scan > results/cifar100/resnet/2con3/cir5_ge_2con3_scan.log
#python main.py --batch_size 128 --circles 2 --backend resnet56_2con3_att --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention scan > results/cifar100/resnet/2con3/cir2_ge_2con3_scan.log
#python main.py --batch_size 128 --circles 8 --backend resnet56_2con3_att --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention scan > results/cifar100/resnet/2con3/cir8_ge_2con3_scan.log
#
##python main.py --batch_size 128 --circles 0 --backend resnet56_2con3_att2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention yes > results/cifar100/resnet/2con3/cir0_ge_2con3_att2.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_2con3_att2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention yes > results/cifar100/resnet/2con3/cir5_ge_2con3_att2.log
#python main.py --batch_size 128 --circles 2 --backend resnet56_2con3_att2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention yes > results/cifar100/resnet/2con3/cir2_ge_2con3_att2.log
#python main.py --batch_size 128 --circles 8 --backend resnet56_2con3_att2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention yes > results/cifar100/resnet/2con3/cir8_ge_2con3_att2.log
#
#python main.py --batch_size 128 --circles 0 --backend resnet56_2con3_att3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention se > results/cifar100/resnet/2con3/cir0_ge_2con3_att3.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_2con3_att3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention se > results/cifar100/resnet/2con3/cir5_ge_2con3_att3.log
#
#python main.py --batch_size 128 --circles 0 --backend resnet56_2con3_att3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention scan > results/cifar100/resnet/2con3/cir0_ge_2con3_att3.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_2con3_att3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy --attention scan > results/cifar100/resnet/2con3/cir5_ge_2con3_att3.log





# 22 Feb
#python main.py --batch_size 128 --circles 0 --backend resnet56_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir0_ge_2con3.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir5_ge_2con3.log
#python main.py --batch_size 128 --circles 8 --backend resnet56_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir8_ge_2con3.log
#python main.py --batch_size 128 --circles 2 --backend resnet56_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir2_ge_2con3.log
#python main.py --batch_size 128 --circles 10 --backend resnet56_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir10_ge_2con3.log
#
#python main.py --batch_size 128 --circles 0 --backend resnet56_2con3_se --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir0_ge_2con3_se.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_2con3_se --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir5_ge_2con3_se.log
#python main.py --batch_size 128 --circles 8 --backend resnet56_2con3_se --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir8_ge_2con3_se.log
#python main.py --batch_size 128 --circles 2 --backend resnet56_2con3_se --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir2_ge_2con3_se.log
#python main.py --batch_size 128 --circles 10 --backend resnet56_2con3_se --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/resnet/2con3/cir10_ge_2con3_se.log





# 21 Feb
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 3.0 --gamma 0.8  > results/cifar100/modelG/2con3/cir5_ge_kd_T3.0_lmbda0.8.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 3.0 --gamma 0.5  > results/cifar100/modelG/2con3/cir5_ge_kd_T3.0_lmbda0.5.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 3.0 --gamma 0.2  > results/cifar100/modelG/2con3/cir5_ge_kd_T3.0_lmbda0.2.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 2.0 --gamma 0.8  > results/cifar100/modelG/2con3/cir5_ge_kd_T2.0_lmbda0.8.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 2.0 --gamma 0.5  > results/cifar100/modelG/2con3/cir5_ge_kd_T2.0_lmbda0.5.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 2.0 --gamma 0.2  > results/cifar100/modelG/2con3/cir5_ge_kd_T2.0_lmbda0.2.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 1.0 --gamma 0.8  > results/cifar100/modelG/2con3/cir5_ge_kd_T1.0_lmbda0.8.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 1.0 --gamma 0.5  > results/cifar100/modelG/2con3/cir5_ge_kd_T1.0_lmbda0.5.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type knowledge_distillation --T 1.0 --gamma 0.2  > results/cifar100/modelG/2con3/cir5_ge_kd_T1.0_lmbda0.2.log



# 20 Feb
#python main.py --batch_size 128 --circles 0 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/2con3/cir0_ge.log
#python main.py --batch_size 128 --circles 2 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/2con3/cir2_ge.log
#python main.py --batch_size 128 --circles 8 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/2con3/cir8_ge.log


# 19 Feb
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/2con3/cir5_se_ge.log
#python main.py --batch_size 128 --circles 0 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/2con3/cir0_se_ge.log
#python main.py --batch_size 128 --circles 2 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/2con3/cir2_se_ge.log
#python main.py --batch_size 128 --circles 8 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/2con3/cir8_se_ge.log

# ---- Variance Test -----
#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> results/cifar100/modelC_h_dp2/no_dropout/cir5_variance_test/2.log
#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> results/cifar100/modelC_h_dp2/no_dropout/cir5_variance_test/3.log
#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> results/cifar100/modelC_h_dp2/no_dropout/cir5_variance_test/4.log
#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> results/cifar100/modelC_h_dp2/no_dropout/cir5_variance_test/5.log
#for circles in 2 8
#do
#    mkdir -p results/cifar100/modelC_h_dp2/no_dropout/cir"$circles"_variance_test
#    for i in 1 2 3 4 5
#    do
#        python main.py --batch_size 128 --circles "$circles" --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 > results/cifar100/modelC_h_dp2/no_dropout/cir"$circles"_variance_test/"$i".log
#    done
#done



#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir5_dp2_dropout1.0_ge.log
#python main.py --batch_size 128 --circles 5 --backend modelG_0con1_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy > results/cifar100/modelG/0con1_2con3/cir5_ge.log
#python main.py --batch_size 128 --circles 5 --backend modelG_2con3 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy >> results/cifar100/modelG/2con3/cir5_ge.log
#python main.py --batch_size 128 --circles 5 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 1 --flops_str 127802496:253664448:547316928  > results/cifar100/modelG/cir5_dp2_ge_ee_lmbda1_2ce.log # with two cross entropy
#python main.py --batch_size 128 --circles 5 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 1 --flops_str 127802496:253664448:547316928  > results/cifar100/modelG/cir5_dp2_ge_ee_lmbda1_2ce.log # with two cross entropy
#python main_cifar.py --batch_size 128 --circles 0 --backend modelG_bl >> results/cifar100/modelG/cir0.log
#python main.py --batch_size 128 --circles 5 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 0 --flops_str 127802496:253664448:547316928  >> results/cifar100/modelG/cir0_dp2_ge_ee_lmbda0.log



# 18 Feb
#python main.py --batch_size 128 --circles 0 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy >> results/cifar100/modelG/cir0_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 5 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy >> results/cifar100/modelG/cir5_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 8 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy >> results/cifar100/modelG/cir8_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 3 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy >> results/cifar100/modelG/cir3_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 10 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type cross_entropy >> results/cifar100/modelG/cir10_dp2_ge_h12.log
#
#python main.py --batch_size 128 --circles 0 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 1 --flops_str 127802496:253664448:547316928  >> results/cifar100/modelG/cir0_dp2_ge_h12_ee.log
#python main.py --batch_size 128 --circles 5 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 1 --flops_str 127802496:253664448:547316928  >> results/cifar100/modelG/cir5_dp2_ge_h12_ee.log
#python main.py --batch_size 128 --circles 8 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 1 --flops_str 127802496:253664448:547316928  >> results/cifar100/modelG/cir8_dp2_ge_h12_ee.log
#python main.py --batch_size 128 --circles 3 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 1 --flops_str 127802496:253664448:547316928  >> results/cifar100/modelG/cir3_dp2_ge_h12_ee.log
#python main.py --batch_size 128 --circles 10 --backend modelG --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --loss_type early_exit --lmbda_e 1 --flops_str 127802496:253664448:547316928  >> results/cifar100/modelG/cir10_dp2_ge_h12_ee.log







# 17 Feb
#python main.py --batch_size 128 --circles 5 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 1 --T 3.0 --gamma 0.9 >> cir5_dp2_ge_kd_T3.0_gamma0.9.log
#
#python main.py --batch_size 128 --circles 0 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir0_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir5_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 6 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir6_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 8 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir8_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 2 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir2_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 3 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir3_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 4 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir4_dp2_ge_h12.log
#python main.py --batch_size 128 --circles 10 --backend resnet56_h12 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 0 >> cir10_dp2_ge_h12.log







# 17 Feb
#python main.py --batch_size 128 --circles 0 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 1 --T 3.0 --gamma 0.9 >> cir0_dp2_ge_kd_T3.0_gamma0.9.log
#python main.py --batch_size 128 --circles 5 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 1 --T 3.0 --gamma 0.9 >> cir5_dp2_ge_kd_T3.0_gamma0.9.log
#python main.py --batch_size 128 --circles 6 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 1 --T 3.0 --gamma 0.9 >> cir6_dp2_ge_kd_T3.0_gamma0.9.log
#python main.py --batch_size 128 --circles 8 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 1 --T 3.0 --gamma 0.9 >> cir8_dp2_ge_kd_T3.0_gamma0.9.log
#python main.py --batch_size 128 --circles 2 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 1 --T 3.0 --gamma 0.9 >> cir2_dp2_ge_kd_T3.0_gamma0.9.log
#python main.py --batch_size 128 --circles 3 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --kd 1 --T 3.0 --gamma 0.9 >> cir3_dp2_ge_kd_T3.0_gamma0.9.log





# 16 Feb
#python main.py --batch_size 128 --circles 0 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir0_adaptive_dp2_ge_dense.log
#python main.py --batch_size 128 --circles 5 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir5_adaptive_dp2_ge_dense.log
#python main.py --batch_size 128 --circles 6 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir6_adaptive_dp2_ge_dense.log
#python main.py --batch_size 128 --circles 8 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir8_adaptive_dp2_ge_dense.log
#python main.py --batch_size 128 --circles 2 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dp2_ge_dense.log
#python main.py --batch_size 128 --circles 3 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir3_adaptive_dp2_ge_dense.log
#python main.py --batch_size 128 --circles 4 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir4_adaptive_dp2_ge_dense.log
#python main.py --batch_size 128 --circles 10 --backend resnet56_dense --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir10_adaptive_dp2_ge_dense.log





# 15 Feb
#python main.py --batch_size 128 --circles 0 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir0_dp2_ge_modelC_dp2.log
#
#python main_cifar.py --batch_size 128 --circles 0 --backend resnet56_bl >> cir0.log
#python main.py --batch_size 128 --circles 0 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir0_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 5 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir5_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 6 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir6_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 8 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir8_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 2 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 3 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir3_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 4 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir4_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 10 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir10_adaptive_dp2_ge.log





# 15 Feb
#python main_cifar.py --batch_size 128 --circles 0 --backend modelC_dp2_bl >> cir0_modelC_dp2.log
#python main.py --batch_size 128 --circles 5 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir5_dp2_ge_modelC_dp2.log
#python main.py --batch_size 128 --circles 6 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir6_dp2_ge_modelC_dp2.log
#python main.py --batch_size 128 --circles 8 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir8_dp2_ge_modelC_dp2.log
#python main.py --batch_size 128 --circles 2 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir2_dp2_ge_modelC_dp2.log
#python main.py --batch_size 128 --circles 3 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir3_dp2_ge_modelC_dp2.log
#python main.py --batch_size 128 --circles 4 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir4_dp2_ge_modelC_dp2.log
#python main.py --batch_size 128 --circles 10 --backend modelC_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir10_dp2_ge_modelC_dp2.log


# 14 Feb 
#python main_cifar.py --batch_size 128 --circles 0 --backend resnet56_bl >> cir0.log
#python main.py --batch_size 128 --circles 0 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir0_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 5 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir5_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 6 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir6_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 8 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir8_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 2 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 3 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir3_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 4 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir4_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 10 --backend resnet56 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir10_adaptive_dp2_ge.log



#python main.py --batch_size 128 --circles 0 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir0_adaptive_dp2_ge.log

# 13 Feb
#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir5_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 2 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 3 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir3_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 4 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir4_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 10 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir10_adaptive_dp2_ge.log


#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir5_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 6 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir6_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 8 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir8_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 2 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 3 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir3_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 4 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir4_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 10 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 >> cir10_adaptive_dp2_ge.log




# 12 Feb
#python main.py --batch_size 128 --circles 6 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir6_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 8 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir8_adaptive_dp2_ge.log
#python main_cifar.py --batch_size 128 --circles 2 --backend prednetE_h >> cir2.log
#python main_cifar.py --batch_size 128 --circles 5 --backend prednetE_h >> cir5.log
#python main_cifar.py --batch_size 128 --circles 0 --backend prednetE_h >> cir0.log
#python main_cifar.py --batch_size 128 --circles 1 --backend prednetE_h >> cir1.log
#python main.py --batch_size 128 --circles 5 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir5_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 2 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 3 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir3_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 4 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir4_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 10 --backend modelC_h_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir10_adaptive_dp2_ge.log


# 11 Feb
#python main.py --batch_size 128 --circles 2 --backend modelC_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir2_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 3 --backend modelC_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir3_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 4 --backend modelC_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir4_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 10 --backend modelC_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir10_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 5 --backend modelC_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 >> cir5_adaptive_dp2_ge.log




#python main.py --batch_size 128 --circles 1 --backend modelC --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --fb 1:1:0 >> cir1_adaptive_dropout0.5_ge_110.log


#python main.py --batch_size 128 --circles 1 --backend modelE_lmbda2 --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0.01 >> cir1_dropout0.5_ge_lmbda0.01.log

# 10 Feb
#python main.py --batch_size 128 --circles 2 --backend modelE_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir2_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 3 --backend modelE_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir3_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 4 --backend modelE_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir4_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 10 --backend modelE_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir10_adaptive_dp2_ge.log
#python main.py --batch_size 128 --circles 5 --backend modelE_dp2 --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir5_adaptive_dp2_ge.log


# 10 Feb
#python main_cifar.py --batch_size 128 --circles 0 --backend prednetF >> cir0.log
#python main_cifar.py --batch_size 128 --circles 1 --backend prednetF >> cir1.log
#python main_cifar.py --batch_size 128 --circles 2 --backend prednetF >> cir2.log
#python main_cifar.py --batch_size 128 --circles 5 --backend prednetF >> cir5.log


# 9 Feb
#python main.py --batch_size 128 --circles 1 --backend modelF --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir1_ge_F.log
#python main.py --batch_size 128 --circles 2 --backend modelE --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir2_ge.log
#python main.py --batch_size 128 --circles 2 --backend modelF --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir2_ge_F.log
#python main.py --batch_size 128 --circles 5 --backend modelE --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir5_ge.log
#python main.py --batch_size 128 --circles 5 --backend modelF --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir5_ge_F.log

# 8 Feb
#python main.py --batch_size 128 --circles 0 --backend modelE --dataset_name cifar100 --max_epoch 300 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir0_ge.log
#python main.py --batch_size 128 --circles 1 --backend modelF --dataset_name cifar100 --adaptive 1 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir1_adaptive_dropout0.5_ge.log

# 6 Feb | modelF
#python main.py --batch_size 128 --circles 1 --backend modelE --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 1.0 --step_all 0 --step_clf 0 --ge 1 --lmbda 0 >> cir1_ge.log

# 5 Feb | prednetE
#python main.py --batch_size 128 --circles 1 --backend modelE --dataset_name cifar100 --adaptive 0 --max_epoch 300 --dropout 0.5 --step_all 0 --step_clf 0 --ge 1 --lmbda 0.01 >> cir1_dropout0.5_ge_lmbda0.01.log
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
