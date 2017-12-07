
# Tested on:
> python 3.5 and TF 1.3

# InfoGAN implementation (https://arxiv.org/abs/1606.03657)

## One regularized variable - Categorical

comment the proper line in main_v0.py
```
$ python main_v0.py
```
After 1st epoch

![epoch 1](cat_epoch_0.png)

After 5th epoch

![epoch 5](cat_epoch_5.png)

After 9th epoch

![epoch 9](cat_epoch_9.png)

Losses

![losses](cat_losses.png)

## One regularized variable - Uniform
comment the proper line in main_v0.py
```
$ python main_v0.py
```
After 1st epoch

![epoch 1](con_epoch_0.png)

After 5th epoch

![epoch 5](con_epoch_5.png)

After 9th epoch

![epoch 9](con_epoch_9.png)

Losses

![losses](con_losses.png)

## Two regularized variables - Categorical viz
comment the proper line in main_v1.py
```
$ python main_v1.py
```
After 10st epoch

![epoch 10](2var_cat_epoch_10.png)

After 50th epoch

![epoch 50](2var_cat_epoch_50.png)

After 79th epoch

![epoch 79](2var_cat_epoch_79.png)

Losses

![losses](2var_cat_losses.png)
## Two regularized variables - Uniform viz
comment the proper line in main_v1.py
```
$ python main_v1.py
```
After 10st epoch

![epoch 10](2var_uni_epoch_10.png)

After 50th epoch

![epoch 50](2var_uni_epoch_50.png)

After 79th epoch

![epoch 79](2var_uni_epoch_79.png)

Losses

![losses](2var_uni_losses.png)
# Links

- (https://arxiv.org/abs/1606.03657)
- (https://github.com/awjuliani/TF-Tutorials/blob/master/InfoGAN-Tutorial.ipynb)
