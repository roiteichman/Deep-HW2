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
    # Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.01
    reg = 0.0001
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

    # Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    # TODO - compare with friends
    wstd = 0.1
    lr_vanilla = 0.05
    lr_momentum = 0.005
    lr_rmsprop = 0.00016
    reg = 0.004
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
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. According to the graphs we can see that the dropout improve the test accuracy and reduce the overfitting as we expected.
we can see that the test loss of dropout=0 is increasing from Iteration=15 while the dropout=0.4 and dropout=0.8 are decreasing.
moreover the test accuracy of the dropout=0.4 and dropout=0.8 are higher than the dropout=0. 

2. We can see that the dropout=0.4 is better than the dropout=0.8 because the dropout=0.8 is too high and the model is
not learning enough, because it is underfitting.
"""

part2_q2 = r"""
**Your answer:**

Yes it is possible.
Cross-entropy loss measures not only whether the predictions are correct
but also how confident the model is about its predictions, 
While accuracy calculates the percentage of correct predictions.
Even if the model correctly classifies more instances,
if its confidence in these predictions is lower than before,
the loss can increase.
For example, if the model shifts from making very confident but incorrect
predictions to making less confident but correct predictions,
the loss might increase despite higher accuracy.

"""

part2_q3 = r"""
**Your answer:**

1. Gradient Descent is an optimization algorithm used to minimize the loss function by iteratively moving
 in the direction defined by the negative of the gradient, which is the steepest descent.
 Back-propagation is a specific algorithm used to calculate the gradients of the loss function with respect
  to the parameters of a neural network. It uses the chain rule to compute these gradients efficiently,
  layer by layer, from the output layer back to the input layer.
 
 2. We will focus in some differences:
 - Batch Size: in Gradient Descent, the batch size is the number of samples used to compute the gradient,
 in contrast to in Stoachastic Gradient Descent, the batch size is 1, which means that the gradient is
 computed for each sample, or subset of samples, in size of a number between 1 and the number of samples.
 - Computational Complexity, Speed And Accuracy Convergence: Gradient Descent is more computationally expensive
 than SGD, because it computes the gradient for the entire dataset, therefore is slower to converge than SGD but also
 more accurate. On the other hand SGD is faster to converge because it computes the gradient for a subset of samples,
 but it is less accurate.
 
  3. SGD used more often than Gradient Descent because it is faster to converge and it is more efficient, making it more
  scalable with large datasets. In addition, the noisy behavior of SGD can help it escape local minima and saddle points, 
  making it is more robust, improve generalization and preventing it from overfitting.
  
  4. 
  - A. The loss of each batch from the forward pass can be written as: $L_i = L(X_i, y_i)$
  Therefore, the total loss over all batches can be written as:
  $L_{\text{total}} = \sum_{i=1}^{B} L_i$ .
  From calculus, we know that the gradient of a sum is the sum of the gradients, thus:
  $\nabla_\theta L_{\text{total}} = \nabla_\theta \sum_{i=1}^{B} L(X_i, y_i) = \sum_{i=1}^{B} \nabla_\theta L(X_i, y_i)$ .
  It means that splitting the data into disjoint batches, do multiple forward passes until all data is exhausted,
  and then do one backward pass on the sum of the losses is equivalent to computing the gradient of the
  loss over the entire dataset as in traditional GD.
  
  - B. We using the memory not only for training the model, but also for storing the data, the model, the gradients,
  the activations and intermediate calculations. The memory error can occur for example when we didn't clearing
  intermediate computational graphs that not needed anymore, or when the intermediate tensors that requiered
  to back propagation are too large to fit in memory.
 
  
"""

part2_q4 = r"""
**Your answer:**

1.
- A. In forward mode AD, we propagate both the value and its derivative through the computational graph.
Typically, forward mode maintains $𝑂(𝑛)$ computation cost but can have high memory complexity if intermediate
derivatives are stored.
    - To reduce memory complexity:
        - initialize $v=x_0$ and the derivative at $v=x_0$ to 1 because $\pderiv{\mat{}}{\mat{X}}X=1$
        - for each operation in the computational graph (for i=1 to n):
            - compute the value of $f$ at $v$ by evaluating the chain of functions: $v = f_i(v)$ .
            - compute the derivative by propagating the derivative forward: $v' = f'_i(v) \cdot v'$ .
    - The memory complexity now is $O(1)$ because in this way we can store only the value and the derivative at each step,
    and not the intermediate derivatives.

- B. In reverse mode AD, we propagate the derivative backward through the computational graph.
Typically, reverse mode maintains $𝑂(𝑛)$ computation cost but can have high memory complexity if intermediate
derivatives are stored.
    - To reduce memory complexity:
        - preform a forward pass and store the intermidiate values $v_i=f_i(v_{i-1})$ for each operation in the computational graph.
        - during the backward pass:
            - initialize the derivative at the output node to $delta=1$
            - for each operation in the computational graph:
                - compute the derivative by propagating the derivative backward: v_i = f_i(v_{i-1})
                - compute $delta = delta \cdot f'_i(v_i)$
    - In this approach recomputes v_i during the backward pass to save memory, and not store the intermediate derivatives,
    therefore the memory complexity now is $O(1)$


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