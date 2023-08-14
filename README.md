# FLPipe
Federated Learning Pipeline

## Requirements
See `requirements.txt`

## Instructions
- Use `make.sh` to generate run script
- Use `make.py` to generate exp script
- Use `process.py` to process exp results
- Hyperparameters can be found in `config.yml` and `process_control()` in `module/hyper.py`

## Examples
 - Generate run script
    ```ruby
    bash make.sh
    ```
 - Generate run script
    ```ruby
    python make.py --mode base
    python make.py --mode fl
    ```
 - Train with MNIST and linear model (FedAvg, 100 horizontally distributed clinets, sychronized with 0.1 active ratio per five local epochs)
    ```ruby
    python train_model_fl.py --control_name MNIST_linear_100-horiz-iid_sync-0.1-5
    ```
 - Test with CIFAR10 and resnet18 model (FedAvg, 100 horizontally distributed clinets, sychronized with 0.1 active ratio per five local epochs)
    ```ruby
    python test_model_fl.py --control_name CIFAR10_resnet18
    ```
<!--  - Process exp results
    ```ruby
    python process.py
    ``` -->

<!-- ## Results
- Learning curves of MNIST
<p align="center">
<img src="/asset/MNIST_Accuracy_mean.png">
</p>


- Learning curves of CIFAR10
<p align="center">
<img src="/asset/CIFAR10_Accuracy_mean.png">
</p>
 -->

## Acknowledgements
*Enmao Diao*
