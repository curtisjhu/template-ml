## Template MNIST

Fundmentally, $ \text{data} \rightarrow \text{model} \rightarrow \text{training}  \rightarrow \text{testing} $

## Getting Started
In your `template-ml` conda env:
```
python pytorch.py
```

### PyTorch Model Iteration 1


MODEL | BATCHES | EPOCHS | ACCURACY
--- | --- | --- | ---
pytorch-model-1 | 32 | 5 | 98.98 %

#### Image of output
<img src="imgs/pytorch-model-1.png" alt="drawing" width="300"/>

Model is a convolutional neural network with max pooling:
```
Using device:cpu
Parameters 13248
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             416
              ReLU-2           [-1, 16, 28, 28]               0
         MaxPool2d-3           [-1, 16, 14, 14]               0
            Conv2d-4           [-1, 32, 14, 14]          12,832
              ReLU-5           [-1, 32, 14, 14]               0
         MaxPool2d-6             [-1, 32, 7, 7]               0
           Flatten-7                 [-1, 1568]               0
           Softmax-8                 [-1, 1568]               0
================================================================
Total params: 13,248
Trainable params: 13,248
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.35
Params size (MB): 0.05
Estimated Total Size (MB): 0.40
```

### PyTorch Model Iteration 2
https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392


