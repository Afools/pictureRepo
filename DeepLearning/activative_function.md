# 激活函数

## 1. Sigmoid 函数

Sigmoid 函数是一个 S 形曲线，输出值在 0 和 1 之间。它的公式为：

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

对其求梯度得到：

$$
f'(x) = f(x)(1 - f(x))
$$

特点是输出值范围有限，容易导致梯度消失问题，尤其是在深层网络中。Sigmoid 函数在神经网络的早期应用中非常流行，但现在通常被其他激活函数所替代。

## 2. Tanh 函数

Tanh 函数是双曲正切函数，输出值在 -1 和 1 之间。它的公式为：

$$
tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

对其求梯度得到：

$$
tanh'(x) = 1 - tanh^2(x)
$$

特点是输出值范围更大，收敛速度比 Sigmoid 函数快，但在输入值较大或较小时，梯度会接近于 0，导致梯度消失问题。

## 3. ReLU 函数

ReLU（Rectified Linear Unit）函数是一个分段线性函数，输出值在 0 和正无穷之间。它的公式为：

$$
f(x) = \max(0, x)
$$

对其求梯度得到：

$$
f'(x) = \begin{cases}
\begin{aligned}
& 1, & x > 0 \\
& 0, & x \leq 0
\end{aligned}
\end{cases}
$$

特点是计算简单，收敛速度快，但在训练过程中可能会出现“死亡 ReLU”现象，即神经元在训练过程中一直输出 0，导致无法更新权重。

## 4. GELU 函数

GELU（Gaussian Error Linear Unit）函数是一个平滑的激活函数，结合了 ReLU 和高斯分布的特点。它的公式为：

$$
GELU(x) = x \cdot P(X \leq x) = x \cdot \Phi(x)
$$

其中 $\Phi(x)$ 是标准正态分布的累积分布函数。对其求梯度得到：

$$
GELU'(x) = \Phi(x) + x \cdot \phi(x)
$$

其中 $\phi(x)$ 是标准正态分布的概率密度函数。GELU 函数在深度学习中表现良好，尤其是在 Transformer 模型中被广泛使用。

## 5. Softmax 函数

Softmax 函数通常用于多分类问题的输出层，将模型的输出转换为概率分布。它的公式为：

$$
softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}
$$

其中 $K$ 是类别数，$x_i$ 是第 $i$ 个类别的得分。Softmax 函数的输出值在 0 和 1 之间，并且所有输出值的和为 1，适合用于多分类问题的概率预测。
对其求梯度得到：

$$
softmax'(x_i) = softmax(x_i) \cdot (1 - softmax(x_i))
$$

## 6. Swish 函数

Swish 函数是一个平滑的激活函数，结合了 Sigmoid 和线性函数的特点。它的公式为：

$$
Swish(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

其中$\sigma(x)$是 sigmoid 激活函数。对其求梯度得到：

$$
Swish'(x) = \sigma(x) + Swish(x) \cdot (1 - \sigma(x))
$$

Swish 在计算上与 ReLU 一样高效，并且在更深的模型上表现出比 ReLU 更好的性能。 函数的曲线是平滑的，并且函数在所有点上都是可微的。这在模型优化过程中很有帮助，被认为是 swish 优于 ReLU 的原因之一。

# 权重初始化

### 1. nn.init.xavier*uniform*

又称为 Glorot 初始化，适用于 sigmoid 和 tanh 激活函数。核心思想是保持每一层的输入和输出的方差相同。

$$
W \sim U\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)
$$

### 2. nn.init.kaiming*uniform*

kaiming 初始化，适用于 ReLU 激活函数。 核心思想是放大输入或输出的方差，以抵消 ReLU 激活函数在负半轴的截断。

$$
W \sim U\left(-\sqrt{\frac{6}{(1+a^2)n_{in}}}, \sqrt{\frac{6}{(1+a^2)n_{in}}}\right)
$$

其中$a$ 是 ReLU 的负斜率，通常取 0.01。

# 损失函数

## 推荐系统损失函数

### 1. BPR Loss

BPR Loss

- 用途：优化推荐系统的排序性能（例如：一个用户更喜欢 A 比 B）。

- 输入数据：三元组 (u, i, j)，表示用户 u 喜欢物品 i（正样本），不喜欢物品 j（负样本）。

- 公式：

  $$
  L_{BPR} = -\log(\sigma(\hat{y}_{ui}-\hat{y}_{uj}))
  $$

  其中 $\hat{y}_{ui}=score(u,i)$, 通常是 dot product。

- 优化目标：让正样本得分高于负样本。

### 2. Dot product loss

- 用途：做二分类预测，即预测 user-item 是否匹配。

- 输入数据：每条 (user, item) 对有明确的 label，0 或 1。

### 3. InfoNCE Loss （Information Noise-Contrastive Estimation）

InfoNCE Losse 是一种**对比学习**损失函数，通常用于无监督学习和自监督学习。它的目标是最大化正样本与负样本之间的相似度差异。

公式：

$$
L_{InfoNCE} = -\log\left(\frac{\exp(sim(\mathbf{z}, \mathbf{z^+})/\tau)}{\exp(sim(\mathbf{z},\mathbf{z^+})/\tau)+\exp(sim(\mathbf{z},\mathbf{z^-})/\tau)}\right)
$$

其中$\mathbf{z}^+,\mathbf{z}^-$分别为正负样本，负样本由正样本随机采样得到，$\tau$为温度参数，$sim(\cdot)$为相似度函数（如点积或余弦相似度）。这样可以最大化正样本之间的相似度，同时最小化正样本与负样本之间的相似度。

# 优化器 optimizer

## 1. Adam

Adam（Adaptive Moment Estimation）是一种常用的**自适应学习率优化器**，结合了 **Momentum（动量）** 和 **RMSProp（均方根传播）** 的优点。

---

### 核心思想：

1. **一阶矩估计（动量）**

   - 跟踪梯度的指数加权平均（类似 Momentum）
   - $ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $

2. **二阶矩估计（梯度平方）**

   - 跟踪梯度平方的指数加权平均（类似 RMSProp）
   - $ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $

3. **偏差修正**

   - 修正前几步估计偏向于 0：
     $$
     \hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
     $$

4. **参数更新**
   $$
   \theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   $$

---

### 超参数：

- $ \eta$：初始学习率
- $ \beta_1 = 0.9 $：动量衰减率
- $ \beta_2 = 0.999 $：梯度平方衰减率
- $ \epsilon = 10^{-8} $：避免除以 0
- $\theta$：学习率

---

### 2. SGD

SGD (Stochastic Gradient Descent) 是最基本的优化器，适用于大多数深度学习任务。
SGD 的核心思想是通过随机选择小批量样本来计算梯度，从而更新模型参数。
学习率更新公式为：

$$
\theta_t = \theta_{t-1} - \eta \cdot g_t
$$

其中 $g_t$ 是当前批次的梯度，$\eta$ 是学习率。
pytorch 中的 SGD 优化器可以通过设置 momentum 参数来实现动量更新。
momentum 的更新公式为：

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t
$$

### 3. AdamW

AdamW 是 Adam 的一种变体，主要用于解决 L2 正则化（权重衰减）的问题。AdamW 在每次参数更新时，将权重衰减项从梯度中分离出来，从而避免了 L2 正则化对学习率的影响。
AdamW 的更新公式为：

$$
\theta_t = \theta_{t-1} - \eta \cdot(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta_{t-1})
$$

这种方法更好的融合 L2 正则化和自适应学习率的优点，通常在训练大型模型时表现更好。**在 Transformer 模型中，AdamW 被广泛使用。**
