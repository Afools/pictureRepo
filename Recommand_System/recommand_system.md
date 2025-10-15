## Job Detail 关键词

sparse LR, FFM, Deep models\
RL, LSTM

## 开源项目

### twitter 推荐算法 [精排算法](https://github.com/twitter/the-algorithm-ml/blob/main/projects/home/recap/README.md)

- 模型：MaskNet
- 输出：用户各种互动方式分数（概率）如下：

```
scored_tweets_model_weight_fav: The probability the user will favorite the Tweet.
scored_tweets_model_weight_retweet: The probability the user will Retweet the Tweet.
scored_tweets_model_weight_reply: The probability the user replies to the Tweet.
scored_tweets_model_weight_good_profile_click: The probability the user opens the Tweet author profile and Likes or replies to a Tweet.
scored_tweets_model_weight_video_playback50: The probability (for a video Tweet) that the user will watch at least half of the video.
scored_tweets_model_weight_reply_engaged_by_author: The probability the user replies to the Tweet and this reply is engaged by the Tweet author.
scored_tweets_model_weight_good_click: The probability the user will click into the conversation of this Tweet and reply or Like a Tweet.
scored_tweets_model_weight_good_click_v2: The probability the user will click into the conversation of this Tweet and stay there for at least 2 minutes.
scored_tweets_model_weight_negative_feedback_v2: The probability the user will react negatively (requesting "show less often" on the Tweet or author, block or mute the Tweet author).
scored_tweets_model_weight_report: The probability the user will click Report Tweet.
```

- 最终得分为所有分数的加权和
  - 根据项目文件，每个分数对最终分数的贡献是近似的，且会定期根据平台参数对权重进行调整

## Transformer 构建

`home/projects/home/recap/model/feature_transform.py`

```python
transformers=[
  InputNonFinite(), # 筛选输入中NaN或无穷大量，将其替换为0
  Log1Abs(),        # 对输入取绝对值加一取对数，再添加原有正负符号
  BatchNorm(),    # 使用了torch.nn.BaatchNorm1d()实现，默认参数affine=False, momentum=0.1
  Clamp(),          # 使用torch.clamp对输入进行截断，所有大于 clip_magnitude和小于-clip_magnitude 的值都截断到最大和最小值，默认最大值为5.0
  cat(x,binary_features)  # 以上过程只对连续特征使用，这里将二元特征并入。
  LayerNorm(),     # 使用torch.nn.LayerNorm()实现，epsilon默认为0，elementwise_affine默认为True。

]

```

## data preprocess

在 twitter 官方实现中仅对数据进行了截断和切片，但是也提供了 down cast, recatify labels 等方法。\
对连续数据截取前 2117，二元数据截取前 59.

从 TFRecord 文件中读取序列化数据，对数据是否进行下采样判断，如果存在下采样，则对数据添加权重

进行数据 shuffle,batching

使用`tf.io.parse_example`和规定的解析字典`segdense.json`进行解析。`parse_fn()`

对数据进行维度整理，同时获得多个任务的标签。`map_output_fn()`核心代码如下

```python
 label_values = tf.squeeze(tf.stack([inputs[label] for label in tasks], axis=1), axis=[-1])
```

最后转为 torch 的 tensor 格式

```python
  def _init_tensor_spec(self):
    def _tensor_spec_to_torch_shape(spec):
      if spec.shape is None:
        return None
      shape = [x if x is not None else -1 for x in spec.shape]
      return torch.Size(shape)

    self.torch_element_spec = tf.nest.map_structure(
      _tensor_spec_to_torch_shape, self._tf_dataset.element_spec
    )
```

# 推荐系统基础

## 指标

### NDCG Normalized Discounted Cumulative Gain

下面是对 **NDCG（Normalized Discounted Cumulative Gain）** 的详细解释，包括含义、核心思想、公式、例子、优点、以及与其他排序指标的对比。

---

### 🧠 一、NDCG 是什么？

**NDCG 是排序模型常用的评价指标，尤其适用于推荐系统、搜索引擎、信息检索等场景。**

它综合衡量两个维度：

1. 推荐结果是否相关（是否是用户喜欢的）
2. 相关的结果是否排得靠前（越靠前越好）

> **核心思想：有用的内容越早出现越好；无用的内容再多也没用。**

---

### 📐 二、NDCG 的计算步骤

#### 步骤 1️⃣：计算 DCG（Discounted Cumulative Gain）

DCG 衡量一个推荐列表的“累积收益”，但随着位置的变后，收益会被“折损”：

$$
\text{DCG@k} = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}
$$

- $rel_i$：位置 $i$ 上 item 的“相关性”分数（可以是 0/1，也可以是 1-5 这种评分）
- $\log_2(i+1)$：位置惩罚函数，位置越靠后，折损越厉害

---

#### 步骤 2️⃣：计算 IDCG（理想 DCG）

这是最理想的推荐列表（相关性高的都排前面）下的 DCG：

$$
\text{IDCG@k} = \text{DCG@k of the ideal sorted list}
$$

---

#### 步骤 3️⃣：归一化，得到最终 NDCG

$$
\text{NDCG@k} = \frac{DCG@k}{IDCG@k}
$$

- NDCG 的值范围是 **0 到 1**，越高越好
- 排名越合理，NDCG 越接近 1

---

### ✅ 四、NDCG 的优点

- ✅ 考虑结果的相关性（rel）和排序位置（log 折损）
- ✅ 可以处理多相关等级（例如 0/1/2/3 分）
- ✅ 对用户行为更真实建模（靠前的内容更有可能被看到）

---

### 🧾 五、适用场景总结：

| 场景                         | 是否适合用 NDCG |
| ---------------------------- | --------------- |
| 信息检索（搜索引擎）         | ✅ 非常适合     |
| 推荐系统（feed 推荐）        | ✅ 适合         |
| 排序类任务（如电商商品排序） | ✅ 适合         |
| 分类任务                     | ❌ 不适合       |
| 只有是否命中的场景           | ❌ Hit@k 更合适 |

---

# 召回

召回是推荐系统的第一个步骤，目的是从物品库中初步提取出用户可能感兴趣的东西（一般是几百个），方便之后进行更进一步的精细化筛选。

### 主要方法

#### ItemCF 基于物品的协同过滤

与基于用户的过滤相对。主要思想是：用户对某一些内容感兴趣，某个内容与这些内容相似，用户可能对相似内容感兴趣。

**Interesting 计算方法**

$$
Interesting = \sum_{i}{Like(user,Item)\times Sim(Item,Item_i)}
$$

其中$Like(user,Item)$表示用户对某个内容的喜欢程度，可以用用户交互方式进行衡量（如点赞，收藏，浏览时间等），$Sim(Item,Item_i)$表示$Item$和$Item_i$的相似程度。最终实际是一种加权求和的形式。

**Similarity 计算方法**

余线相似度

$$
Sim(i,j)=\frac{\sum_{v\in V}{Like(v,i)\cdot Like(v,j)}}{\sqrt{\sum_{u\in W_1}{Like^2(u,i)}\cdot\sqrt{\sum_{u\in W_2}{Like^2(u,j)}}}}
$$

其中$W_1$是对 $i$ 有互动的用户，$W_2$是对 $j$ 有互动的用户，$V=W_1\cap W_2$ 表示对两者都有互动的。

如果不考虑喜欢程度，只考虑是否交互，可以简化成如下形式：

$$
Sim(i,j)=\frac{||V||}{\sqrt{||W_1||\cdot ||W_2||}}
$$

**步骤**

1. 离线计算\
   建立用户-> 物品索引\
   建立物品-> 物品索引
2. 线上计算\
   对给定用户，根据索引得到最近感兴趣/有交互的物品列表，长度为 K\
   根据物品列表得到相似物品表\
   对每个物品的相似物品，选择其中前 n 个最相似的物品，得到$nK$个物品\
   选择其中 Interesting 得分最高的前 100 个物品召回.

**ItemCF 缺陷**

ItemCF 在计算物品相似度时，没有考虑到用户-物品交互的社群效应。如若干物品分享到一个 QQ 群中，这些物品有更大概率被该 QQ 群的用户群体交互，但这些物品不一定有高相似性。

**Swing 计算法**

为了减少社群对物品相似度的影响，Swing 方法引入用户相似度对余弦相似度进行调整，方法如下：

1. 计算用户相似度$overlap(u_1,u_2)=|I_1\cap I_2|$ ，$I_1,I_2$是用户 1，用户 2 喜欢的物品。
2. 计算相似度
   $$
   sim(I_1,I_2)=\sum_{v_1\in V}\sum_{v_2\in V}{\frac{1}{\alpha +overlap(v_1,v_2)}}
   $$
   其中$\alpha$为超参数

#### UserCF 基于用户的协同过滤

与 ItemCF 类似，不同点如下

**用户相似度计算**

$$
Sim(u_1,u_2)=\frac{\sum_{l\in I}\frac{1}{\log{(1+n_l)}}}{\sqrt{||I_1||\cdot||I_2||}}
$$

其中$I$为两个用户同时喜欢的物品，$n$为喜欢该物品的总人数，用于削减高热度物品对相似度的影响。

**实际应用中 ItemCF 往往比 UserCF 效果好，原因如下**

- 用户兴趣变化快，用户之间的相似度不稳定
- 用户交互行为稀疏，导致用户相似度计算不准确

#### 矩阵补全方法

这是一个早期推荐系统使用的推荐方法。

对用户和物品分别构建一个矩阵 P、Q，用$P\cdot Q$拟合数据。numpy 实现如下

损失函数：

$$
L=\sum_{ij}{(R_{ij}-P_i\cdot Q_j^T)^2}+\lambda(||P||^2+||Q||^2)\\
e_{ij}=R_{ij}-P_iQ_j^T
$$

梯度下降：

$$
P_i=P_i+lr\cdot (e_{ij}-\lambda P_i)\\
Q_j=Q_j+lr\cdot (e_{ij}-\lambda Q_j)
$$

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 示例评分矩阵 R（0 表示缺失）
R = torch.tensor([
    [5.0, 3.0, 0.0],
    [4.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 5.0],
    [0.0, 0.0, 4.0],
])

num_users, num_items = R.shape
latent_dim = 2  # 潜在因子维度

# 掩码矩阵，指示哪些评分是已知的
mask = (R > 0).float()

# 可学习参数：用户矩阵 P，物品矩阵 Q
P = torch.randn(num_users, latent_dim, requires_grad=True)
Q = torch.randn(num_items, latent_dim, requires_grad=True)

# 优化器
optimizer = optim.Adam([P, Q], lr=0.01)
loss_fn = nn.MSELoss()

# 训练
for epoch in range(1000):
    optimizer.zero_grad()

    # 预测评分
    R_hat = torch.matmul(P, Q.t())

    # 只在已知评分上计算 MSE
    loss = loss_fn(R_hat * mask, R * mask)

    # 加正则项（防止过拟合）
    reg = 0.01 * (P.norm() + Q.norm())
    total_loss = loss + reg

    total_loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.item():.4f}")

# 最终预测评分矩阵
R_pred = torch.matmul(P, Q.t()).detach()
print("\n预测评分矩阵 R_pred:")
print(R_pred)

```

#### 双塔模型

- 双塔模型增加了用户和物品本身的特征，如离散特征包括：用户性别、城市、感兴趣的话题，物品的类型、品牌等；连续特征包括：用户的浏览时间、年龄、活跃程度、消费金额，物品的价格、销量、浏览量等。

- 对用户 ID 和离散特征进行 embedding，连续特征进行归一化或分桶处理。物品的特征处理方式与用户相同。

- 所有特征经过 embedding 后，分别输入到两个神经网络中，分别计算用户和物品的向量表示。两个网络的结构可以相同，也可以不同。

- 对用户和物品的向量表示计算余弦相似度，得到用户和物品的匹配度。

**训练方法**
双塔模型的训练目标是让模型学会区分正样本和负样本。正样本是用户和物品的真实交互，负样本是随机生成的用户和物品的组合。训练方法可以使用对比损失函数，如 InfoNCE Loss 或 BPR Loss。

根据负样本的采样方法，双塔模型可以分为三类：

- pointwise：独立看待每个正样本和负样本。\
  将正样本和负样本的匹配度作为二分类问题进行训练。对正样本，希望 sim(a,b)接近 1；对负样本，希望 sim(a,b)接近 0。通常正负样本比例为 1:2 或 1:3。
- pairwise：对每个正样本取一个负样本。\
  鼓励$sim(a,b^+)>sim(a,b^-)$，即正样本的匹配度大于负样本的匹配度。通常正负样本比例为 1:1。
  因此设置损失函数如下：
  $$
  L_{pairwise} = \max(0, \alpha - sim(a,b^+) + sim(a,b^-))
  $$
  其中$\alpha$是一个超参数，表示正样本和负样本的间隔。\
  在 BPR 算法中，取消了$\alpha$，直接使用 sigmoid 函数进行训练。损失函数如下：
  $$
  L_{BPR} = -\log(\sigma(sim(a,b^+) - sim(a,b^-)))
  $$
- listwise：每个正样本对应多个负样本列表。\
  List-wise 排序是将整个 item 序列看作一个样本，通过直接优化信息检索的评价方法和定义损失函数两种方法来实现。List-wise 损失函数因为训练数据的制作难，训练速度慢，在线推理速度慢等多种原因，尽管用的还比较少，但是因为更注重排序结果整体的最优性，所以也是目前很多推荐系统正在做的事情。\
  ListMLE loss 是 listwise 排序中最常用的损失函数之一。其核心思想是将整个 item 序列看作一个样本，通过直接优化信息检索的评价方法来实现。其损失函数如下：
  $$
  ListMLE = -\sum_{k=1}^{n}\log(\frac{e^{s_k}}{\sum_{j=k}^{n}e^{s_j}})
  $$
  其中$s_k$是第 k 个 item 的得分，$n$是 item 的总数。\
  ListNet Loss 是以序列中每个 item 的 Top1 概率为基础的损失函数。其损失函数如下：
  $$
  P_s(j)=\frac{\exp{s_j}}{\sum_{i=0}^n \exp{s_i}}\\
  ListNet = -\sum_{i=1}^{n}P_y(i)\log(P_s(i))
  $$
  以上是 ListNet 的交叉熵表示法。其中$P_y(i)$是序列中第 i 个 item 的真实概率，$P_s(i)$是序列中第 i 个 item 的预测概率。

### 负样本采样

负样本采样是推荐系统中非常重要的一步。负样本的质量直接影响模型的训练效果和性能。常用的负样本采样方法有以下几种：

1. 简单负样本：

- **全物品简单随机采样**：因为正样本稀疏，负样本约等于数据集中所有的样本，所以可以随机采样。此方法需要考虑样本点击率，均匀采样会对冷门物品不公平，非均匀采样会对热门物品不公平，**一般使用点击率$^{0.75}$作为负样本的采样概率**。
- **batch 内负样本**：每个 batch 内，一个 user 和一个存在交互的 item 作为一个正样本，有 n 个用户就有 n 个正样本。一个 user 和其他 user 交互的 item 组成负样本，n 个用户能组成$n(n-1)$个负样本。
- 在 batch 内采样中，物品出现在 batch 内的概率与物品的点击率成正比。因此，热门物品的采样概率更高。应用中通过对预估概率增加偏置来纠正这种偏差。比如将用户 u 对物品 i 感兴趣的概率由$cos(u,i)$改为$cos(u,i)-log(p_i)$，其中$p$正比于物品 i 的点击率。

2. 困难负样本:\
   困难负样本指的是用户可能感兴趣，但不够感兴趣的样本。比如粗排后淘汰的样本，精排分数靠后的样本。\
   `需要注意，曝光但没点击的物品不要作为负样本，因为曝光的物品可能是用户感兴趣的物品，只是有更感兴趣的物品吸引了用户注意。在实践中将曝光但没点击的物品作为负样本会导致模型表现下降。但是，曝光但没点击的物品可以作为后续排序的负样本。`\
   训练数据通常按比例混合简单负样本和困难负样本如(1:1)

### 线上召回和更新

在实际应用中，双塔模型需要实时对用户进行推荐。

#### 1. 在线召回

对物品塔进行离线训练，用户塔进行在线训练。物品塔的参数定期更新，用户塔的参数实时更新。因为物品的特征相对固定，而用户的兴趣和行为相对多变。

- 物品塔进行训练后，离线计算得到物品的向量表示，将物品的向量存储在向量数据库中，存储格式如<Item_embedding, Item_id>。在数据库中建立索引，方便进行最邻近查找。
- 用户塔进行训练后，在线计算得到用户的向量表示，在数据库中进行最邻近查找，得到用户最可能感兴趣的物品。

#### 2. 更新方式

1. **全量更新**：如在每天凌晨，用昨天全天的数据 random_shuffle 后再昨天模型的基础上进行 1 epoch 训练。当天部署新的用户模型和物品向量。
2. **增量更新**：如每小时用过去一小时的数据对 embedding 过程进行更新，用以实时给用户推荐最近感兴趣的物品。**注意：增量更新不对神经网络部份进行更新。**因为小时级别的数据量太小，可能会导致模型过拟合。

### 自监督学习

双塔模型具有很强的头部效应，即模型对热门物品的推荐效果很好，但对冷门物品（长尾物品、低曝光物品）的推荐效果较差。为了解决这个问题，可以使用自监督学习的方法来增强模型的泛化能力。

通过数据增强，更好的学习长尾物品的向量表示，提高模型对冷门物品和新物品的推荐效果。

### 曝光过滤方法 Bloom Filter

当一个物品出现再用户眼中后，之后的召回要将其排除，防止用户看到重复的物品。在小红书、抖音等产品都有使用。
一般使用 Bloom Filter 来实现。

Bloom Filter 是一种空间效率高的概率型数据结构，可以用来测试一个元素是否在一个集合中。它的特点是：

- 可以快速判断一个元素是否又可能在集合中（存在误判）
- 如果判断为否，则一定不在；如果判断为是，则很可能在集合中（存在误判）
- 通常使用多个哈希函数来减少误判率
- 适合存储大量数据，且只需要判断是否存在的场景
  误判概率计算如下：

  $$
  \Pr(\text{false positive}) = (1 - e^{-kn/m})^k
  $$

  其中 k 是哈希函数的数量，n 是元素的数量，m 是位数组的大小。

- Slide Bloom Filter\
  由于 Bloom Filter 是一个静态数据结构，不能动态添加或删除元素，因此需要使用 Slide Bloom Filter 来处理过时的元素。
  - 维护多个 Bloom Filter，每个 Bloom Filter 只存储一段时间内的元素。
  - 查询时，检查所有 Bloom Filter，直到找到一个 Bloom Filter 返回 false positive。
  - 定期清除过时的 Bloom Filter。

# 排序

## Tricks

### 预估偏差矫正

在负样本采样中，需要对负样本进行降采样，避免负样本过多导致模型训练不稳定。而在点击率预估时，正负样本实际分布的差异会导致点击率预估出现偏差。需要进行偏差矫正，矫正方法如下：

- 假设正样本数量为 $N_{pos}$，负样本数量为 $N_{neg}$。在采样时对负样本进行降采样，得到负样本数量为$\alpha N_{neg}$，其中$\alpha\in(0,1)$为降采样率。
- 预估点击率为 $$CTR_{pred}=\frac{N_{pos}}{N_{pos}+\alpha N_{neg}}$$实际点击率为 $$CTR_{real}=\frac{N_{pos}}{N_{pos}+N_{neg}}$$
- 因此可以通过以下公式进行偏差矫正：
  $$
  CTR_{real}=\frac{\alpha CTR_{pred}}{(1-CTR_{pred})+\alpha CTR_{pred}}
  $$

## Multi-gate Mixture-of-Experts (MMoE)

MMoE 是一种多路复用的混合专家模型，主要用于推荐系统中的多任务学习。MMoE 通过多个专家网络和门控机制来实现对不同任务的建模。其核心思想是将多个专家网络与门控网络结合起来，以便在每个任务上选择最相关的专家进行计算，从而提高模型的性能和效率。
![](https://raw.githubusercontent.com/Afools/pictureRepo/main/Recommand_System/MMoE.png)
模型通过通常 4 个或 8 个专家模型和若干个权重控制模型来实现对不同任务的建模。每个专家模型都是一个独立的神经网络，负责处理特定的任务。门控网络根据输入数据和任务类型来选择最相关的专家进行计算，从而提高模型的性能和效率。
![](https://raw.githubusercontent.com/Afools/pictureRepo/main/Recommand_System/MMoE_output.png)

**极化**
在训练过程中，MMoE 可能会出现极化现象，即某些专家网络的权重过高，而其他专家网络的权重过低。这可能导致模型在某些任务上的性能下降。为了解决这个问题，可以使用 dropout 等正则化方法来防止极化现象的发生。

如 dropout，将权重模型的输出结果每一项都有一定概率被置为 0，迫使模型学习到更鲁棒的特征表示。

## 预估分数融合

得到多任务的预估分数后，需要对这些分数进行融合，以便得到最终的排序结果。

1. 线性加权融合：对每个任务的预估分数乘以一个权重系数，然后相加得到最终的排序结果。权重系数可以通过交叉验证等方法进行调优。
2. 点击率加权和：将点击率和其他预估分数的加权和相乘，得到最终的排序结果：
   $$score = p_{click} \cdot (1+w_i\cdot p_{like}+\cdots)$$
   其中 $w_i$ 是第 i 个任务的权重系数，$p_{like}=\#点赞/\#点击$, $p_{click}=\#点击/\#曝光$。
3. 其他方法：如 youtube 融分公式如下：
   $$ score = (1+w*1\cdot p*{time})^{\alpha*1}\cdot(1+w_2\cdot p*{like})^{\alpha_2}\cdots$$

### 视频完播率指标转换

视频完播率指标存在对长视频、短视频的不公平现象，视频时长越长，其完播率天然越低，因此需要对视频完播率进行转换，防止对长视频的不公平。具体方法如下：

1. 用视频时长拟合完播率曲线，得到一个函数 $f(t)$，表示视频时长为 $t$ 时的完播率。
2. 使用函数 $f(t)$ 对完播率进行转换，得到最终的完播率指标：
   $$
      p_{final} = \frac{p_{raw}}{f(t)}
   $$

## 粗排

粗排目的是给召回得到的几千个物品进行排序，得到前几百（多是 100）个物品。粗排模型通常使用双塔模型或三塔模型。

### 三塔模型

三塔模型介于双塔模型和精排深度神经网络之间，  拥有相对双塔模型更高的准确率和相对精排更好的性能。其与双塔模型的区别在于，三塔模型增加了一个交叉塔，其输入是统计特征和交叉特征。统计特征是用户和物品的统计特征，如用户的浏览时间、物品的浏览量等。交叉特征是用户和物品的交叉特征，如用户对物品的点击率、用户对物品的浏览时间等。
