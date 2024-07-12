r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1.
A. The Jacobian tensor represents how each element in $Y$ changes with respect to each element in $X$.
$Y$ has a shape of $(N, out Features)$ and $X$ has a shape of $(N, in Features)$.
$W$ has a shape of $(out Features, in Features)$, the layer calculates $Y = XW^T$.
Therefore, based on the definition of the Jacobian tensor,
the shape of $\pderiv{\mat{Y}}{\mat{X}}$ is $(N, out Features, N, in Features) = (64, 512, 64, 1024)$.

B. Each element in the output vector $𝑌$ is formed by a linear combination of a single row
in the input matrix $𝑋$, which has $𝑁$ rows.
When differentiating the output element with respect to different input samples, only 
$1/𝑁$ of the elements in $\pderiv{\mat{Y}}{\mat{X}}$ are non-zero. Consequently, the Jacobian tensor is sparse.

C. Instead of materializing the Jacobian tensor, it will be much more efficient to use the chain rule, as we learned in class,
to compute: $∂x = \pderiv{\mat{L}}{\mat{X}} = \pderiv{\mat{L}}{\mat{Y}} \pderiv{\mat{Y}}{\mat{X}}$

2. 
A. The Jacobian tensor represents how each element in $Y$ changes with respect to each element in $W$.
$Y$ has a shape of $(N, out Features)$ and $X$ has a shape of $(N, in Features)$.
$W$ has a shape of $(out Features, in Features)$, the layer calculates $Y = XW^T$.
Therefore, based on the definition of the Jacobian tensor,
the shape of $\pderiv{\mat{Y}}{\mat{X}}$ is $(N, out Features, out Features, in Features) = (64, 512, 512, 1024)$.

B. TODO - compare to friends: 
based on ChatGPT answer:
The Jacobian tensor $\frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$ is not sparse by definition.
Each element $y_{ij}$ in $\mathbf{Y}$ depends linearly on the row $i$ of the input $\mathbf{X}$
and the entire weight matrix $\mathbf{W}$. Specifically, each element $y_{ij}$ can be expressed as:
$y_{ij} = \sum_{k=1}^{1024} x_{ik} w_{jk}$ .
The partial derivative of $y_{ij}$ with respect to $w_{jk}$ is $x_{ik}$,
and hence the Jacobian is filled with these values. There are no zero elements by definition.


our answer:
Each element in the output vector $𝑌$ is formed by a linear combination of a single column
in the weight matrix $W$, which has $in Features$ columns.
When differentiating the output element with respect to different weights, only 
$1/in Features$ of the elements in $\pderiv{\mat{Y}}{\mat{W}}$ are non-zero. Consequently, the Jacobian tensor is sparse.

C. We don't need to materialize the Jacobian tensor, it will be much more efficient to use the chain rule:
$∂W = \pderiv{\mat{L}}{\mat{W}} = \pderiv{\mat{L}}{\mat{Y}} \pderiv{\mat{Y}}{\mat{W}}$
"""

part1_q2 = r"""
**Your answer:**

Theoretically, it's possible to compute gradients of the loss with respect to weights in neural networks
without relying on back-propagation and the chain rule.
This would involve manually calculating gradients for each parameter,
disregarding the efficient layer-by-layer propagation used in back-propagation.
However, this approach would be impractical for deep networks with many layers and parameters.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""