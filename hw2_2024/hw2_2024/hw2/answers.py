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
    wstd = 0.001
    lr_vanilla = 0.02
    lr_momentum = 0.35
    lr_rmsprop = 0.00019
    reg = 0.0018
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
    wstd = 0.002
    lr = 0.03
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
            - initialize the derivative at the output node to $δ=1$
            - for each operation in the computational graph:
                - compute the derivative by propagating the derivative backward: $v_i = f_i(v_{i-1})$
                - compute $δ = δ \cdot f'_i(v_i)$
    - In this approach recomputes v_i during the backward pass to save memory, and not store the intermediate derivatives,
    therefore the memory complexity now is $O(1)$.

2. These techniques can be generalized for arbitrary computational graphs by applying similar principles of storing
minimal intermediate values and recomputing them as necessary. For example, by using a checkpointing strategy to store
only the critical nodes in the computational graph and recomputing intermediate values on demand, we can reduce memory
complexity in both forward and reverse mode AD. Specifically, in forward mode AD, store only the current value and derivative
and in backward mode AD, store critical nodes and recompute intermediate values on demand.

3. When trying to applied these techniques to deep architectures like ResNets or VGGs that have many layers and parameters,
the memory complexity can be very high and the memory error can occur. By applying these techniques, that helps to balance
the trade-off between memory and computation, we can reduce the memory complexity and prevent memory errors, making it
possible to train very deep networks more efficiently, without exceeding memory limits.

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

1. High Optimization Error: Optimization error occurs when the model fails to find the best parameters that minimize the training loss.
To reduce it we can use GD variants like SGD, Adam, RMSprop, etc,
because they can help to reduce the variance of the gradient estimation, and therefore reduce the optimization error.
Another option is to use a learning rate scheduler, that can help to reduce the learning rate during training,
and therefore reduce the optimization error.

2. High Generalization Error: Generalization error refers to the difference between the training error and the test error. A high generalization error indicates that while the model performs well on the training data, it performs poorly on unseen test data, suggesting overfitting.
To reduce it we can use regularization techniques like L1, L2, dropout, etc,
that can help to reduce the model complexity and prevent overfitting.
Another option is to use early stopping, that can help to stop the training when the validation error is not improving,
and therefore prevent overfitting.
Moreover we can use data augmentation techniques, that increasing the size of the training set,
or batch normalization, that can help to stabilize the training process.
We have learned them both in lectures 4-5, and they both can improve the generalization.  
In CNNs we can increase the receptive field, that allows the model to capture more context 
and improve its ability to generalize. Another option is to use techniques like max pooling, or mean pooling, 
that can help to reduce the spatial dimension of the input, and therefore reduce the model complexity and prevent overfitting.
 
3. High Approximation Error: Approximation error occurs when the chosen model or hypothesis class is too simple to capture the relationships in the data. This suggets underfitting.
To reduce it we can use a more complex model, like a deeper model,
or different architecture, like CNN, RNN, etc, that can help to learn more complex patterns.
Moreover, we can use a different activation function, like ReLU, LeakyReLU, etc, that can help to learn more complex patterns.
In addition, we can use a different optimizer, like Adam, RMSprop, etc, that can help to learn more complex patterns.
Other option is to use boosting techniques, like AdaBoost, Gradient Boosting, etc, that can help to learn more complex patterns. 
In CNNs we can increase the receptive field, that allows the model to capture more context
and improve its ability to generalize.

"""

part3_q2 = r"""
**Your answer:**

Scenario: COVID-19 Screening Test

High False Positive Rate:
- In an area where COVID-19 prevalence is low, a high FPR means that many people who do not have COVID-19 are being
incorrectly classified as positive by the screening test.
- Reasons for a high FPR could include:
    - Inaccurate test results due to very sensitivity test that trying to ensure no cases are missed.
    - Precautionary principle, where the test is designed to be overly cautious to avoid missing any cases.
    - Incorrect interpretation of test results due to human error.

High False Negative Rate:
- In an area where COVID-19 prevalence is high, a high FNR means that many people who have COVID-19 are being
incorrectly classified as negative by the screening test.
- Reasons for a high FNR could include:
    - Speed over accuracy, where the test is designed to be fast and easy to administer, but may miss some cases.
    - Inaccurate test results due to low sensitivity test.
    - Inadequate sample collection or handling leading to false negatives.
    - Limited resources or testing capacity leading to false negatives.

"""

part3_q3 = r"""
**Your answer:**

1. Since the disease will eventually show non-lethal symptoms leading to diagnosis and treatment,
the goal is to avoid unnecessary expensive and risky tests for healthy patients.
This means accepting a higher rate of false negatives because these cases will be caught later when symptoms develop.
Therefore, we need a high threshold where the true positive rate (TPR) is reasonable and the false positive rate (FPR) is low.
This will ensure that the test is not overly sensitive and does not produce too many false positives.

2. Missing a diagnosis can be fatal, so the model must be highly sensitive to ensure early detection.
False positives, while costly and risky, are preferable to missing cases that could result in death.
Therefore, we need a low threshold where the TPR is high even if the FPR is high.
This will ensure that the test is highly sensitive and does not miss any cases, even if it produces many false positives.

"""


part3_q4 = r"""
**Your answer:**

MLP might not be the best choice for train on sequential data because:
- MLPs treat each input independently and do not inherently capture the sequential nature of the data.
In the case of text, where the order of words is crucial for understanding the meaning and context,
an MLP does not account for the relationships between words across different positions in the sequence.
- MLPs have a fixed input size, which can be problematic for sequences of varying lengths.
For example, in the case of text, sentences can have different lengths, and an MLP would require padding or truncating
to fit all inputs to a fixed size.
- MLPs do not have memory, so they cannot remember past inputs or context from previous time steps. This is crucial for
sequential data where the current output depends on previous inputs. For example, in language modeling, the prediction
of the next word depends on the words that came before it.


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
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.001
    weight_decay = 0
    momentum = 0.99
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1. As we learned in the lecture, the number of parameter is determine by the following formula:
$N_{\text{parameters}} = (F \cdot F \cdot K + 1)\cdot L)$
where:
- $F$ is the size of the filter.
- $K$ is the number of input channels.
- $L$ is the number of output channels.
+1 because the bias is counted as a parameter.
Therefore, the number of parameters in the first layer is:
    - So if we have two convolutional layers with 3x3 kernel, 256 input channel and 256 output channel, the number of parameters is: 
    $N_{\text{parameters}} = (3 \cdot 3 \cdot 256 + 1) \cdot 256) + (3 \cdot 3 \cdot 256 + 1) \cdot 256) = 1,180,160$
    - on the other hand, if we 256 input channel and 256 output channel, and we use in bottleneck layer 1x1 kernel,
    to reduce to a layer of 64 output channel, then making 3x3 kernel to 64 output channel, and then 1x1 kernel to 256 output channel,
    the number of parameters is:
    $N_{\text{parameters}} = (1 \cdot 1 \cdot 256 + 1) \cdot 64) + (3 \cdot 3 \cdot 64 + 1) \cdot 64) + (1 \cdot 1 \cdot 64 + 1) \cdot 256) = 70,400$

2. The Number of floating point operations required to compute an output of a convolutional layer is determined by the following formula:
$FLOPs = (F \cdot F \cdot K \cdot L \cdot H \cdot W)$
where:
- $F$ is the size of the filter.
- $K$ is the number of input channels.
- $L$ is the number of output channels.
- $H$ is the height of the input.
- $W$ is the width of the input.
Therefore, the number of FLOPs in the first layer is:
    - So if we have two convolutional layers with 3x3 kernel, 256 input channel and 256 output channel, and the input image is HxW,
    the number of FLOPs is: $FLOPs = (3 \cdot 3 \cdot 256 \cdot 256 \cdot H \cdot W) + (3 \cdot 3 \cdot 256 \cdot 256 \cdot H \cdot W) = 1,572,864 \cdot H \cdot W$
    - on the other hand, if we 256 input channel and 256 output channel, and we use in bottleneck layer 1x1 kernel,
    to reduce to a layer of 64 output channel, then making 3x3 kernel to 64 output channel, and then 1x1 kernel to 256 output channel,
    the number of FLOPs is: $FLOPs = (1 \cdot 1 \cdot 256 \cdot 64 \cdot H \cdot W) + (3 \cdot 3 \cdot 64 \cdot 64 \cdot H \cdot W) + (1 \cdot 1 \cdot 64 \cdot 256 \cdot H \cdot W) = 1,179,648 \cdot H \cdot W$

3. $Regular Block:$

$Spatially:$ Both convolutions are 3x3, so each can combine information from a 3x3 neighborhood, enhancing spatial feature extraction across multiple layers (the receptive field is larger allowing for better spatial feature extraction).

$Across Feature Maps:$ Both 3x3 convolutions maintain the full 256-channel depth throughout the block. This means that each convolution can combine information from all input channels, preserving and enhancing channel relationships. Each output channel is influenced by 256 input channels.

$Bottleneck Block:$

$Spatially:$ The 1x1 convolutions before and after the 3x3 convolution do not contribute to spatial feature extraction (smaller effective receptive field), as they only operate on individual pixels.

$Across Feature Maps:$ The sequence in the block enables flexible feature map integration: initially, feature reduction concentrates on key features, the 3x3 convolution spatially combines them, and the final expansion enhances different channel relationships.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**
5.1
1. Deeper networks can learn more complex and abstract representations of the input data, resulting in better accuracy.
On the other hand deeper networks are also harder to train due to various reasons such as prone to overfitting, longer training times and vanishing/ exploding gradients.
In experiment 1.1, the model with depth=4 performed the best overall. This is probably due to the fact that the model with depth=4 had a good tradeoff between the ability to learn more complex features, and was not too deep to suffer from overfitting or vanishing gradients.

2. Yes, the network of L=16 were not trainable. This is possibly due to exploding/vanishing gradients. We can resolve this problem partially by:

   a) Employing advanced weight initialization methods  to ensure that the initial weights are set in a way that mitigates the vanishing/exploding gradient problem.
   
   b) Incorporating Batch Normalization layers in the network can reduce co-variate shift, stabilize training, and accelerate the process by normalizing the inputs to each layer and maintaining a consistent distribution of activations.
"""

part5_q2 = r"""
**Your answer:**

We used most of the same settings, but change the pooling kernel size according to the different depths.
From the graphs we observed that for shallower depths (smaller L's) the convergence was faster, while for narrower networks (for a fixed depth) the convergence was longer.
We can see from the different depths that as the depth increased we got better test accuracy, and compared to experiment 1.1 we got the best accuracy for L=8 rather than L=4.
This Means that with the configuration in this experiment, and the wider network we used was able to capture more complex features.

"""

part5_q3 = r"""
**Your answer:**

We used pooling kernel size that was efficient for shallow networks like previous experiments.
Similar to the previous experiments, the shallower networks converged faster than the deeper ones.
We can see that the test accuracy of the deeper networks is higher than the shallower ones, but the accuracy of all the depth is similar (75%-80%).
An interesting observation is that adding more filters didin't improve the test accuracy significantly compared to other experiments.

"""

part5_q4 = r"""
**Your answer:**

We can see in this experiment that for L=16 the network is trainable, while in experiment 1.1 it wasn't.
This is probably due to the fact that in this experiment we used the resnet model - meaning we added skip connections.
We can conclude that by using the resnet model, we can train deeper networks and thus maybe capture more complex features.
Also, we can conclude from the graphs in this experiment that the depth and the width of the network had pretty much the same effect on the test accuracy (although the different results on the train set).
Lastly, while comparing the results of this experiments to the results in experiment 1.3 we can see that adding more filters did improve the test accuaracy.


"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

1.
In the first picture, the model correctly identified all objects with accurate boundaries but failed to categorize them properly.
In the second picture, the detection quality was lower, merging multiple objects (such as a cat and a dog) into one.
However, the categorization was somtimes correct.

2.
Possible reasons for the poor performance of the model:
- The model may not have been trained on a sufficiently diverse or large dataset, leading to poor generalization to new images.
- The dataset might be imbalanced, with some classes underrepresented.
- The architecture or hyperparameters used might not be optimal for the given task.
Possible methods that can help to resolve the poor performance:
- Collect more labeled data, especially for underrepresented classes.
- Ensure an even distribution of all classes in the training data.
- Experiment with different architectures and hyperparameters to find the best-performing model configuration (as we did in the experiments).

3.
To attack YOLO using the PGD method, we create small changes in the input image that reduce the model's accuracy. 
This involves calculating the gradient of the loss function with respect to the image and adding noise in the direction of the gradient, while keeping the changes small enough to be unnoticeable but still effective in confusing the model.



"""


part6_q3 = r"""
**Your answer:**
Image 1:
The model does not detect correctly the ducks, it detects multiple ducks together as a fire hydrant.
The models accuracy for the detection is 68%.
The pitfall that is causing the model to detect the ducks as a fire hydrant is probably due to the fact that the ducks are aligned some are partially hidden by others, causing the model to miss important features.
Meaning the model is encountering the occlusion pitfall.

Image 2:
The model does not identify the man riding the bicycle. The pitfall that the model is encountering is blurring, meaning the objects are in motion and appear blurry.
Therefor the model does not detect any object in the picture.

Image 3:
The model does not identify the dog correctly in the picture, it actually detects the dog as a cow.
This is because the model is encountering the textured background pitfall, meaning the model has trouble identifies the object due to the background.



"""

part6_bonus = r"""
**Your answer:**

Image 3:
With the removal of the background, and coloring it as white, the model was able to detect the dog correctly.
This is because the textured background pitfall was removed, and the model was able to focus on the object itself.
The model's accuracy for the detection was 100%.

"""