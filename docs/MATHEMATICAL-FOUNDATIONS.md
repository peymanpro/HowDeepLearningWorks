# Mathematical Foundations

This note follows the calculations implemented in the core library. All vectors below are treated as column vectors.

## 1. Dense-layer forward pass

Suppose the input has $n$ features and the layer has $m$ neurons:

$$
x \in \mathbb{R}^{n}, \quad
W \in \mathbb{R}^{m \times n}, \quad
b \in \mathbb{R}^{m}
$$

The affine transformation and activation are:

$$
z = Wx+b \in \mathbb{R}^{m}, \qquad
a = \phi(z) \in \mathbb{R}^{m}
$$

The shape of each value explains why the multiplication is valid:

| Value | Shape |
|---|---:|
| $W$ | $m \times n$ |
| $x$ | $n \times 1$ |
| $Wx$ | $m \times 1$ |
| $b$, $z$, $a$ | $m \times 1$ |

The implementation stores weights as a matrix with **one row per output neuron** and **one column per input feature**.

## 2. Backpropagation through one layer

Let $L$ be the scalar loss. Backpropagation receives the upstream gradient:

$$
g = \frac{\partial L}{\partial a} \in \mathbb{R}^{m}
$$

Since the activation is applied element-wise, the chain rule gives:

$$
\delta = \frac{\partial L}{\partial z}
= g \odot \phi'(z)
$$

Here, $\odot$ means element-wise multiplication. The gradients for the parameters and the input are:

$$
\frac{\partial L}{\partial W} = \delta x^T
$$

$$
\frac{\partial L}{\partial b} = \delta
$$

$$
\frac{\partial L}{\partial x} = W^T \delta
$$

The shapes are consistent:

- $\delta x^T$ has shape $m \times n$, matching $W$.
- The bias gradient has shape $m$, matching $b$.
- $W^T\delta$ has shape $n$, matching $x$.

The input gradient is what allows a previous layer to continue the chain rule backward. All layers must calculate their gradients before their parameters are updated for that training example; otherwise the backward pass could use a mixture of old and new weights.

## 3. Binary Cross-Entropy and Sigmoid

For a predicted probability $p$ and target $y \in [0,1]$, Binary Cross-Entropy is:

$$
L(p,y) = -\left(y\log p + (1-y)\log(1-p)\right)
$$

Its derivative with respect to the prediction is:

$$
\frac{\partial L}{\partial p}
= -\frac{y}{p} + \frac{1-y}{1-p}
$$

The implementation's **BinaryCrossEntropy.Derivative** returns this derivative with respect to **the probability**, not with respect to the pre-activation logit.

For a Sigmoid output $p=\sigma(z)$,

$$
\frac{\partial p}{\partial z}=p(1-p)
$$

Therefore, away from the numerical clipping boundaries:

$$
\frac{\partial L}{\partial z}
= \frac{\partial L}{\partial p}
  \frac{\partial p}{\partial z}
= p-y
$$

The current code uses the general chain-rule path: the loss provides $\partial L/\partial p$, then the output layer multiplies it by the Sigmoid derivative. This keeps the responsibilities of loss and activation explicit for learning purposes.

To prevent logarithms of zero, Binary Cross-Entropy clamps the probability into a small interval inside $(0,1)$. The formulas above describe the ordinary interior case; behavior at the clipping boundary is a numerical safeguard rather than a fully differentiable clipping operation.

## 4. Gradient descent

With learning rate $\eta > 0$, a parameter is updated in the direction opposite its gradient:

$$
W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial L}{\partial W}
$$

$$
b_{\text{new}} = b_{\text{old}} - \eta \frac{\partial L}{\partial b}
$$

This is stochastic gradient descent in the sense that the demonstration updates parameters after individual training examples. It does not currently implement mini-batches, momentum, or Adam.

## 5. Numerical gradient checking

Backpropagation derives gradients analytically. A finite-difference approximation provides an independent check for a parameter $\theta$:

$$
\frac{\partial L}{\partial \theta}
\approx
\frac{L(\theta+\varepsilon)-L(\theta-\varepsilon)}
{2\varepsilon}
$$

The central difference uses loss values on both sides of the parameter and typically has lower truncation error than a one-sided difference. It is still an approximation, so the comparison uses a tolerance rather than requiring exact equality.

The console verification compares analytical and numerical gradients for all **172 weights** in its five-layer example. The xUnit suite additionally checks weight and bias gradients in a smaller network with non-saturated Sigmoid output and ReLU activations away from their non-differentiable point.

A gradient check can fail or become misleading if an activation crosses a non-smooth boundary during the perturbation, if the loss is numerically saturated, or if $\varepsilon$ is poorly chosen. Test inputs and initial parameters are therefore selected so the finite-difference comparison examines a smooth local region.

## 6. Implementation boundaries

The layer stores the most recent input and pre-activation so that **Backward** can use the cache created by **Forward**. This is pedagogically simple, but it also means the same layer instance is stateful: run one forward pass and its backward pass before reusing it for another training example. The current design is intended for single-example educational execution, not concurrent or batched training.

The matrix and vector classes use straightforward managed C# loops. The implementation prioritizes visible arithmetic and testability over throughput, SIMD kernels, GPU execution, or the broader features expected from a production ML framework.
