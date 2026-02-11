#!/usr/bin/env bash

python main.py -data banknote -root ../data -net banknote_mlp -weights_path ./networks/weights/best_banknote_mlp.pt -results_path ./results
python main.py -data cifar10 -root ../data -net cifar10_resnet18 -weights_path ./networks/weights/best_cifar_resnet18.pt -results_path ./results
python main.py -data mnist -root ../data -net mnist_mlp -weights_path ./networks/weights/best_mnist_mlp.pt -results_path ./results
python main.py -data mnist -root ../data -net mnist_cnn -weights_path ./networks/weights/best_mnist_cnn.pt -results_path ./results