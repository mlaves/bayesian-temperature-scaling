#!/bin/sh

rm -rf __pycache__/
BS=256

python -u train.py --batch_size=${BS} --epochs=500 --model=resnet18 --dataset=cifar10 | tee `date '+%Y-%m-%d_%H-%M-%S'`_resnet18.log
python -u train.py --batch_size=${BS} --epochs=500 --model=resnet101 --dataset=cifar100 | tee `date '+%Y-%m-%d_%H-%M-%S'`_resnet101.log
python -u train.py --batch_size=${BS} --epochs=500 --model=densenet169 --dataset=cifar100 | tee `date '+%Y-%m-%d_%H-%M-%S'`_densenet169.log
# python -u train.py --batch_size=${BS} --epochs=500 --model=densenet121 --label_smoothing=0.0 --dataset=cifar10 | tee `date '+%Y-%m-%d_%H-%M-%S'`_densenet121_no_cp.log
# python -u train.py --batch_size=${BS} --epochs=500 --model=densenet121 --label_smoothing=0.1 --dataset=cifar10 | tee `date '+%Y-%m-%d_%H-%M-%S'`_densenet121_cp.log
