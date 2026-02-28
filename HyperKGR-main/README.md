# HyperKGR


Dataset can be found from 

inductive   https://github.com/LARS-research/RED-GNN/tree/main/inductive/data

transductive   https://github.com/LARS-research/RED-GNN/tree/main/transductive/data

## Instructions

A quick instruction is given for readers to reproduce the whole process.



Requirements 

- pytorch  1.9.1+cu102
- torch_scatter 2.0.9



For transductive reasoning

    cd transductive
    python -W ignore train.py --data_path=data/WN18RR



For inductive reasoning

    cd inductive
    python -W ignore train.py --data_path=data/WN18RR_v1



### Data splition in transductive setting

We follow the rule mining methods, i.e., [Neural-LP](https://github.com/fanyangxyz/Neural-LP) and [DRUM](https://github.com/alisadeghian/DRUM), to randomly split triplets in the original `train.txt` file into two files `facts.txt` and `train.txt` with ratio 3:1. This step is to make sure that the query triplets will not be leaked in the fact triplets used in HyperKGR. Empirically, increasing the ratio of facts, e.g. from 3:1 to 4:1, will lead to better performance.



## Citation

If you find this repository useful in your research, please kindly cite our paper.

```
@inproceedings{liu2025hyperkgr,
  title={HyperKGR: Knowledge Graph Reasoning in Hyperbolic Space with Graph Neural Network Encoding Symbolic Path},
  author={Lihui Liu},
  booktitle={EMNLP},
  year={2025}
}
```

⏺ Transductive vs Inductive 再解释

  打个比方：

  Transductive（直推式）—— 开卷考试，考的都是书上的人物

  - 训练时：给你一个社交网络，里面有 1000 个人，你知道其中一部分关系
  - 测试时：还是这 1000 个人，问你缺失的关系是什么
  - 核心：人都是认识的，只是关系不全

  Inductive（归纳式）—— 换一批人重新考

  - 训练时：给你 A 城市 1000
  个人的社交网络，学会"朋友的同事往往也是朋友"这类规律
  - 测试时：给你 B 城市 500
  个全新的人，这些人训练时一个都没见过，让你用学到的规律去推理
  - 核心：考的是你有没有学到通用的关系模式，而不是记住了谁是谁

  所以你的理解是对的：本质就是测试时的实体在训练时有没有见过。

  topk 和 layers 是什么

  layers：GNN 的层数，也就是消息传递几轮。

  - 1 层 = 只看直接邻居（1 跳）
  - 3 层 = 能看到 3 跳远的邻居
  - 层数越多，能推理的路径越长，但计算量也越大

  topk：每层采样保留多少个节点。

  - 假设 1 层之后扩展出了 5000 个邻居节点，但 topk=1000
  意味着只保留得分最高的 1000 个，剩下的丢掉
  - 作用是控制计算量，不然几层下来节点数爆炸
  - 不同数据集图的大小不同，所以 topk 不一样：family 小图用 100，fb15k-237
  大图用 2000

  简单说：layers 决定看多远，topk 决定每步看多少个节点。


  ⏺ 整体流程（以 transductive sample 版本为例）

  1. 数据准备

  原始知识图谱：(张三, 朋友, 李四), (李四, 同事, 王五), ...

  - 读入 entities.txt、relations.txt 建立 ID 映射
  - 读入三元组，自动添加反向边：(李四, 朋友_inv, 张三)，这样关系数变成
  2*n_rel+1（+1 是自环）
  - 每个 epoch 按 fact_ratio（比如 0.96）随机切分：96% 当建图的事实，4%
  当训练的查询
  - 用稀疏矩阵 csr_matrix 建立邻接结构，方便快速查邻居

  2. 训练一个 batch 的流程

  假设一个查询是 (张三, 朋友, ?)：

  输入：一批查询 [(张三, 朋友), (王五, 同事), ...]   共 B 个
  目标：预测每个查询的尾实体

  第 0 步：初始化

  活跃节点 = [张三, 王五, ...]       每个查询只有头实体自己
  隐藏表示 = 全零向量               每个节点一个 hidden_dim 维向量

  第 1~N 层：GNN 消息传递（核心循环）

  每一层做 4 件事：

  ┌─────────────────────────────────────────────────────┐
  │ ① 扩展邻居                                          │
  │    活跃节点 → 查稀疏矩阵 → 拿到所有相邻的边           │
  │    比如张三的邻居：(张三, 朋友, 李四), (张三, 同学, 赵六)│
  │                                                      │
  │ ② 计算注意力（在欧氏切空间中）                         │
  │    对每条边算重要性分数：                              │
  │    α = sigmoid(W · ReLU(Ws·源节点 + Wr·边关系 + Wqr·查询关系))│
  │    比如：查的是"朋友"关系，那"朋友"边的权重 > "同学"边    │
  │                                                      │
  │ ③ 双曲空间消息传递                                    │
  │    切空间 → expmap0 → 双曲空间                        │
  │    双曲空间中做 TransE 平移：head ⊕ relation           │
  │    （⊕ 是 Möbius 加法，不是普通加法）                   │
  │    双曲空间 → logmap0 → 回到切空间                     │
  │    消息 = α × 平移结果                                │
  │                                                      │
  │ ④ 聚合 + 更新                                        │
  │    scatter_sum：同一目标节点的消息求和                  │
  │    线性变换 → expmap0 → 激活函数 → logmap0             │
  │    GRU 门控：融合本层结果和上层记忆                     │
  │                                                      │
  │ ⑤ 节点采样（仅 sample 版本）                          │
  │    对新扩展出的节点打分，只保留 topk 个                 │
  │    训练时用 Gumbel-Softmax（可导的近似采样）            │
  │    推理时用普通 Softmax                                │
  └─────────────────────────────────────────────────────┘

  每层结束后，活跃节点集扩大（但被 topk 限制住），hidden 表示更新。

  直觉理解：像水波一样从头实体一圈一圈往外扩散，每一层扩一跳，同时收集路径上
  的信息。

  3. 打分 + 计算损失

  经过 N 层后：
      每个活跃节点有一个 hidden 向量
      ↓
      W_final 线性层 → 每个节点得到一个标量分数
      ↓
      铺到 [B, 全部实体数] 的矩阵上（没访问到的实体得分为 0）
      ↓
      交叉熵损失：正确答案的分数应该最高
      ↓
      反向传播，Adam 优化器更新参数

  4. 评估

  对每个测试查询 (h, r, ?)：
      ① 模型给所有实体打分
      ② 过滤掉已知正确答案（避免惩罚正确预测）
      ③ 看真正答案排第几名
      ④ 汇总指标：MRR（平均倒数排名）、Hits@1、Hits@10

  5. 一张图总结

  查询 (h, r, ?)
        │
        ▼
     [h] ─── 第1层 ──→ [h, 邻居1, 邻居2, ...] ─── topk采样 ──→ 保留k个
                                │
                          第2层 ──→ [更多节点...] ─── topk采样 ──→ 保留k个
                                │
                          第N层 ──→ [最终活跃节点集]
                                │
                        W_final 打分
                                │
                      每个实体得到分数
                                │
                    loss = 交叉熵(分数, 正确答案)

  关键点：整个过程中，消息传递不是在欧氏空间做普通加法，而是在 Poincaré
  球（双曲空间）中做 Möbius
  加法。双曲空间的好处是天然适合表示树状/层次结构，而知识图谱中很多关系（如
  is_a、part_of）恰好是层次性的。

✻ Churned for 39s

❯ 再给一个举例                                                              
                                                                         
⏺ 举个完整的例子                                                            
                                                                         
  假设我们有一个小型知识图谱：                                              
   
  (北京, 首都_of, 中国)                                                     
  (上海, 城市_in, 中国)                                     
  (中国, 位于, 亚洲)                                                        
  (日本, 位于, 亚洲)                                        
  (东京, 首都_of, 日本)
  (大阪, 城市_in, 日本)

  现在问模型：(东京, 城市_in, ?) → 应该答 日本

  ---
  第 0 步：数据预处理

  自动添加反向边，图变成双向的：

  原始边：(北京, 首都_of, 中国)
  反向边：(中国, 首都_of_inv, 北京)

  原始边：(东京, 首都_of, 日本)
  反向边：(日本, 首都_of_inv, 东京)

  ... 每条边都加反向

  再给每个节点加自环：(东京, 自环, 东京)

  ---
  第 1 层：从东京出发，看 1 跳邻居

  活跃节点：{东京}

  查邻居，找到 3 条边：
      (东京, 自环, 东京)
      (东京, 首都_of, 日本)
      (日本, 首都_of_inv, 东京)   ← 反向边，方向是日本→东京

  活跃节点扩展为：{东京, 日本}

  计算注意力（我们查的是 城市_in 关系）：

  边 (东京, 首都_of, 日本)：
      α = sigmoid(W · ReLU(Ws·东京的表示 + Wr·首都_of的表示 +
  Wqr·城市_in的表示))
      "首都_of" 和 "城市_in" 语义接近 → α 比较高，比如 0.8

  边 (东京, 自环, 东京)：
      自环只是保留自身信息 → α 比较低，比如 0.3

  双曲空间消息传递（以"东京→日本"这条边为例）：

  切空间（普通向量）          双曲空间（Poincaré 球内）          切空间

  东京的表示 hs ──expmap0──→  hs'  ─┐
                                     ├─ Möbius加法 ─→ project ──logmap0──→
  消息向量
  首都_of表示 hr ──expmap0──→ hr' ─┘

                          （在弯曲空间里做平移，
                           不是简单的向量相加）

  日本节点收到的消息 = 0.8 × 消息向量

  聚合 + GRU 更新：

  日本的新表示 = GRU(本层聚合结果, 上层记忆)
  东京的新表示 = GRU(本层聚合结果, 上层记忆)

  假设 topk=3，当前只有 2 个节点，不需要裁剪。

  ---
  第 2 层：从 {东京, 日本} 出发，再扩一跳

  活跃节点：{东京, 日本}

  东京的邻居：东京自己、日本（已有）
  日本的邻居：
      (日本, 位于, 亚洲)           ← 新节点！
      (日本, 首都_of_inv, 东京)    ← 已有
      (日本, 城市_in_inv, 大阪)    ← 新节点！
      (日本, 自环, 日本)

  活跃节点扩展为：{东京, 日本, 亚洲, 大阪}

  注意力（还是在查 城市_in）：

  (日本, 城市_in_inv, 大阪)：
      "城市_in_inv" 是 "城市_in" 的反向 → 和查询关系直接相关 → α 很高，比如
  0.9

  (日本, 位于, 亚洲)：
      "位于" 和 "城市_in" 有一定关系但不那么直接 → α 中等，比如 0.5

  假设 topk=3，新扩展了亚洲和大阪 2 个节点，加上之前的东京、日本共 4
  个，需要砍掉 1 个：

  对新节点打分：
      亚洲：0.5 分
      大阪：0.9 分

  保留 top-1 新节点 → 保留大阪，丢掉亚洲
  最终活跃节点：{东京, 日本, 大阪}

  ---
  第 3 层（最后一层）：再扩一跳

  类似过程，可能扩展出中国等节点，继续聚合消息。

  ---
  最终打分

  经过 3 层后，每个活跃节点都有一个融合了多跳路径信息的表示向量

  W_final 打分：
      东京：-1.2
      日本：3.8    ← 最高分！
      大阪：0.5
      其他未访问到的实体：0

  模型预测：日本 ✓

  ---
  为什么双曲空间有用？

  欧氏空间（平面）：              双曲空间（Poincaré 圆盘）：

      亚洲                              亚洲（圆心附近）
     /    \                            /         \
    中国   日本                      中国          日本
    / \    / \                      / \          / \
  北京 上海 东京 大阪              北京 上海    东京 大阪（靠近边缘）

  知识图谱天然有层次结构（上位概念→下位概念）。双曲空间越靠近边缘，空间增长
  越快，天然能用少量维度就表达树状层次关系。欧氏空间做同样的事需要更高的维度
  。


  sample 版本                                                               
                                                            
  Transductive：                                                            
                                                            
  # family
  cd sample/transductive && python3 train.py --data_path ./data/family/
  --train --topk 100 --layers 8 --gpu 0

  # umls
  cd sample/transductive && python3 train.py --data_path ./data/umls/
  --train --topk 100 --layers 5 --gpu 0

  # WN18RR
  cd sample/transductive && python3 train.py --data_path ./data/WN18RR/
  --train --topk 1000 --layers 8 --gpu 0

  # fb15k-237
  cd sample/transductive && python3 train.py --data_path ./data/fb15k-237/
  --train --topk 2000 --layers 7 --remove_1hop_edges --gpu 0

  # nell
  cd sample/transductive && python3 train.py --data_path ./data/nell/
  --train --topk 2000 --layers 6 --gpu 0

  # YAGO
  cd sample/transductive && python3 train.py --data_path ./data/YAGO/
  --train --topk 1000 --layers 8 --gpu 0

  Inductive（只需指定数据路径，配置全自动）：

  cd sample/inductive && python3 train.py --data_path ./data/WN18RR_v1
  cd sample/inductive && python3 train.py --data_path ./data/fb237_v1
  cd sample/inductive && python3 train.py --data_path ./data/nell_v1
  # v2, v3, v4 替换即可

  not_sample 版本

  # transductive
  cd not_sample/transductive && python -W ignore train.py
  --data_path=data/WN18RR

  # inductive
  cd not_sample/inductive && python -W ignore train.py
  --data_path=data/WN18RR_v1

  评估（加载已有模型）

  cd sample/transductive && python3 train.py --data_path ./data/WN18RR/
  --eval --topk 1000 --layers 8 --gpu 0 --weight
  ./data/WN18RR/8-layers-best.pt