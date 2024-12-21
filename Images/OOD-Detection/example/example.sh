python -m train lrn-8-16-32-resnet-20 cifar10 -e 100 -s --opt=adam --lr=0.01 --weight-decay=1e-4
python -m ood_detection --model lrn-8-16-32-resnet-20 --dataset cifar10 --num_to_avg 10