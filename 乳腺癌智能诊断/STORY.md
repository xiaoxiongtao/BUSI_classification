# STORY.md · 《乳腺癌智能诊断》

## ① 用户意图对齐

- **目标受众**：高校人工智能 / 生物医学工程方向的本科生与研究生，在"深度学习与医学影像分析"课堂上听讲。已具备 CNN、反向传播、梯度下降等基础概念，但尚未系统学习残差网络与 Transformer 在视觉任务中的应用。
- **核心目标**：讲完后学生要相信——"同一份 780 张的小样本医学数据上，模型架构的归纳偏置比参数量更决定成败"；要记住 ResNet 残差连接与 ViT 自注意力两条主干的结构与数学表达；要能做——按任务书独立跑通一次训练与评估，并读懂混淆矩阵与 F1 的含义。
- **PPT 长度**：38 页（含封面、目录、7 个章节扉页、结束页）。Hero 页配额 8 页，占 21%，符合 20–30%。
- **视觉调性**：学术严谨 / 蓝白克制 / 证据驱动 / 结构清晰 / 医学真实感。
- **内容边界**：
  - 必讲：BUSI 数据与任务定义；ResNet 残差块与整体架构；ViT 分块嵌入、位置编码、多头自注意力、Encoder Block；交叉熵损失、类别权重、标签平滑；本项目的真实实验结果与结论分析；课堂实战任务。
  - 不讲：分割/检测任务、3D 超声、临床注册与合规、其他 Transformer 变体（Swin/DeiT 仅在展望中一句带过）。
  - 禁碰：虚构临床结论、宣称可用于真实诊断、编造论文数据。所有数字来自本地真实训练产物 `outputs/logs/ppt_model_summary.json`。

## ② 页面布局骨架

### 章节划分（目录 7 章 → 7 个 section 扉页，编号 01–07 连续）

| 章 | 编号 | 标题 | 扉页页码 | 内容页 |
| :-- | :-- | :-- | :-- | :-- |
| 一 | 01 | 临床背景与数据 | P3 | P4–P6 |
| 二 | 02 | 技术路线与数据处理 | P7 | P8–P10 |
| 三 | 03 | ResNet：深度残差网络 | P11 | P12–P16 |
| 四 | 04 | ViT：视觉 Transformer | P17 | P18–P22 |
| 五 | 05 | 损失函数与优化策略 | P23 | P24–P26 |
| 六 | 06 | 实验结果与分析 | P27 | P28–P33 |
| 七 | 07 | 课堂实战任务 | P34 | P35–P37 |

### Hero 页定位（8 页，21%）

P1 封面、P3 章扉01、P11 章扉03、P17 章扉04、P19 ViT 整体框架、P29 结果总览巨型数字、P34 章扉07、P38 结束页。
任意两 Hero 之间至少间隔 1 个 Supporting 页 ✅。

### rhythm 曲线

P1 peak｜P2 transition｜P3 transition｜P4 valley｜P5 peak｜P6 valley｜P7 transition｜P8 valley｜P9 valley｜P10 peak｜P11 transition｜P12 valley｜P13 peak｜P14 valley｜P15 valley｜P16 peak｜P17 transition｜P18 valley｜P19 peak｜P20 valley｜P21 peak｜P22 valley｜P23 transition｜P24 peak｜P25 valley｜P26 valley｜P27 transition｜P28 valley｜P29 peak｜P30 valley｜P31 peak｜P32 valley｜P33 peak｜P34 transition｜P35 valley｜P36 valley｜P37 peak｜P38 peak

无连续 ≥3 页 valley（P8–P9 后有 P10 peak；P24–P26：P25、P26 两 valley 后接 P27 transition；P28 一 valley 后 P29 peak；P35–P36 两 valley 后 P37 peak）✅

### 非对称版式预算

非对称版式共 24 页 / 38 页 = 63% ≥ 40% ✅
对称版式仅 2 页：P2 目录（编号条）、P29（居中巨型数字）。其余章节扉页按"全屏视觉"处理为仪式段，不计入常规对称预算。

## ③ 页面大纲

| # | title | type | role | rhythm | layout | visual | visual_role | density | anti_pattern | description |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | 乳腺癌智能诊断 | cover | hero | peak | 全屏视觉+大标题 | L1 cover_ultrasound.png 全幅底图 | atmosphere | 字数约60/图1/留白30% | 禁止在封面放 200×70 小装饰图 | 封面交代课程标题、英文长标题、项目来源与主讲信息 |
| 2 | 课程导览 | catalog | transition | transition | 编号条目录 | L3 编号方块 | atmosphere | 字数约150/图0/留白35% | 禁止圆球编号、英文比中文抢眼 | 7 章目录，让学生先建立路线图 |
| 3 | 01 临床背景与数据 | section | hero | transition | 全屏视觉+大标题 | L1 超声影像半透明背景 | atmosphere | 字数约70/图1/留白32% | 禁止四卡片预览 | 第一章扉页 |
| 4 | 为什么需要智能诊断 | content | supporting | valley | 左标题+右内容 | Chart(柱状图)+L2 sample_malignant.png | evidence | 字数约260/图2/留白18% | 禁止等宽卡片横排 | 780 张数据只有 210 例恶性——样本稀缺与医生经验依赖，是 AI 介入的切入点 |
| 5 | BUSI 数据集 | content | supporting | peak | 上大图+下方卡片 | L1 dataset_samples.png | anchor | 字数约240/图1/留白15% | 禁止 50:50 等分双栏 | 437/210/133 的类别分布决定了必须做分层划分与类别加权 |
| 6 | 任务定义与评价指标 | content | supporting | valley | 非对称双栏 60:40 | Diagram(指标公式)+Table | evidence | 字数约250/图1/留白18% | 禁止只列公式不解释临床含义 | 恶性召回率比总准确率更重要——漏诊代价远高于误诊 |
| 7 | 02 技术路线与数据处理 | section | hero | transition | 全屏视觉+大标题 | L1 流程线稿 | atmosphere | 字数约70/图1/留白32% | 禁止铺满正文段落 | 第二章扉页 |
| 8 | 整体技术路线 | content | supporting | valley | 非对称版式（流程图） | L1 Diagram(7 步流水线) | evidence | 字数约230/图1/留白16% | 禁止把流程拆成等宽七卡片 | 从读图到出报告共 7 步，ResNet 与 ViT 共用同一条流水线保证公平 |
| 9 | 数据预处理与增强 | content | supporting | valley | 左大图+右侧文字 | L2 增强示意 SVG + CodeBlock | evidence | 字数约250/图2/留白18% | 禁止只贴代码不解释目的 | 随机裁剪翻转旋转色彩抖动——用人工多样性补足 780 张的不足 |
| 10 | 训练策略与超参数 | content | supporting | peak | 左标题+右内容 | Table(超参表)+KPI 三联数 | evidence | 字数约240/图0/留白18% | 禁止把超参表缩到角落 | 50 epoch / AdamW / StepLR / 早停 15——早停是本实验最关键的防过拟合闸门 |
| 11 | 03 ResNet：深度残差网络 | section | hero | transition | 全屏视觉+大标题 | L1 残差连接线稿 | atmosphere | 字数约70/图1/留白32% | 禁止 PART 03 字样泛滥 | 第三章扉页 |
| 12 | 网络加深的困境 | content | supporting | valley | 非对称双栏 65:35 | Chart(退化曲线示意)+L3 | evidence | 字数约250/图1/留白18% | 禁止只说"梯度消失"四个字 | 56 层训练误差反而高于 20 层——这不是过拟合，是优化失败 |
| 13 | 残差学习：核心思想 | content | supporting | peak | 巨型数字+洞察 | L1 Diagram(残差块)+巨型公式 | anchor | 字数约230/图1/留白20% | 禁止把公式塞进图表角落 | H(x)=F(x)+x——让网络只需学残差，学不动时退化为恒等映射 |
| 14 | 两种残差块 | content | supporting | valley | 非对称双栏 | L1 SVG(BasicBlock/Bottleneck) | evidence | 字数约240/图1/留白16% | 禁止左右等分双栏 | BasicBlock 双 3×3 与 Bottleneck 1×1-3×3-1×1 的取舍 |
| 15 | ResNet 家族整体架构 | content | supporting | valley | 左大图+右侧文字 | L1 SVG(五阶段架构) | anchor | 字数约240/图1/留白16% | 禁止把架构图缩成 240×160 | stem→4 stage→GAP→FC，四种深度只是 block 数与类型不同 |
| 16 | 代码实现与迁移学习 | content | supporting | peak | 左标题+右内容 | CodeBlock + L2 | evidence | 字数约230/图1/留白18% | 禁止整页代码无讲解 | 20 行核心 forward + ImageNet 预训练迁移，是把 11M 参数用在小数据上的关键 |
| 17 | 04 ViT：视觉 Transformer | section | hero | transition | 全屏视觉+大标题 | L1 patch 网格线稿 | atmosphere | 字数约70/图1/留白32% | 禁止与封面雷同 | 第四章扉页 |
| 18 | 从卷积到注意力 | content | supporting | valley | 非对称双栏 60:40 | Diagram(感受野对比) | evidence | 字数约250/图1/留白18% | 禁止只写"全局建模"不解释 | 卷积局部共享权重，注意力一次看全图——代价是要自己学归纳偏置 |
| 19 | ViT 整体框架 | content | hero | peak | 全幅图+骑线文字 | L1 SVG(ViT 全流程) | anchor | 字数约240/图1/留白18% | 禁止标题下塞小图 | 224×224 → 196 个 patch token → 12 层 Encoder → CLS token 出分类 |
| 20 | 分块嵌入与位置编码 | content | supporting | valley | 上大图+下方卡片 | L1 SVG(patch 切分)+Math | evidence | 字数约240/图2/留白16% | 禁止把 196 个 patch 画成抽象色块 | 一个 Conv2d(3,768,k=16,s=16) 就完成了分块嵌入 |
| 21 | 多头自注意力 | content | hero | peak | 左大图+右侧文字 | L1 SVG(QKV)+Math 公式 | anchor | 字数约240/图2/留白18% | 禁止只放公式不给直觉 | 12 个头并行关注不同区域，QK^T/√d 的缩放防止 softmax 饱和 |
| 22 | Encoder Block 与变体 | content | supporting | valley | 非对称双栏 | L1 SVG(Encoder)+Table | evidence | 字数约240/图2/留白16% | 禁止多张深蓝实色卡 | Pre-Norm + 双重残差；B/16 到 L/32 四档变体差异 |
| 23 | 05 损失函数与优化策略 | section | hero | transition | 全屏视觉+大标题 | L1 损失曲面线稿 | atmosphere | 字数约70/图1/留白32% | 禁止满版蓝红金 | 第五章扉页 |
| 24 | 交叉熵损失 | content | supporting | peak | 巨型数字+洞察 | Math(Softmax+CE)+Diagram | anchor | 字数约240/图1/留白18% | 禁止只写公式不举数值例子 | 一个具体三分类数值例子，让学生看到 loss 从 1.20 降到 0.34 意味着什么 |
| 25 | 类别不平衡与加权 | content | supporting | valley | 非对称双栏 60:40 | Table(权重)+L3 | evidence | 字数约240/图1/留白18% | 禁止只说"数据不平衡" | 恶性权重 1.71、正常 2.66——少数类被漏判时损失惩罚翻倍 |
| 26 | 标签平滑与优化调度 | content | supporting | valley | 左标题+右内容 | Math+Table(优化器) | evidence | 字数约250/图0/留白18% | 禁止把优化器写成一句话 | ε=0.1 平滑 + AdamW 解耦权重衰减 + 每 15 epoch 学习率 ×0.1 |
| 27 | 06 实验结果与分析 | section | hero | transition | 全屏视觉+大标题 | L1 曲线线稿 | atmosphere | 字数约70/图1/留白32% | 禁止与第三章扉页雷同 | 第六章扉页 |
| 28 | 实验设置与公平对比 | content | supporting | valley | 左标题+右内容 | Table(设置)+L3 | evidence | 字数约240/图0/留白18% | 禁止把设置表放在页面中央 60% | 同划分、同增强、同优化器，唯一变量是架构 |
| 29 | 六模型测试集结果 | content | hero | peak | 居中金句/巨型数字 | 大数字 89.8% + Table | anchor | 字数约220/图0/留白20% | 禁止等宽卡片横排 | ResNet50 拿下 89.8% 准确率与 0.890 macro-F1；ViT 两版仅约 53%——参数量最大却垫底 |
| 30 | 最优模型逐类表现 | content | supporting | valley | 左大图+右侧文字 | L1 cm_resnet50.png + Table | evidence | 字数约240/图1/留白16% | 禁止把混淆矩阵缩到 240×160 | 恶性召回 87.5%、特异度 96.5%——漏诊 4 例，仍是临床最敏感的部分 |
| 31 | 训练曲线对比 | content | supporting | peak | 非对称双栏 | L1 curves_resnet50.png + curves_vit_b16.png | anchor | 字数约230/图2/留白15% | 禁止两图等大并排 | ResNet50 验证损失稳步下降并早停于 49 轮；ViT-B/16 验证损失长期高于训练损失 |
| 32 | 为什么 ViT 失利 | content | supporting | valley | 非对称双栏 65:35 | Diagram(归纳偏置对比) | evidence | 字数约260/图1/留白16% | 禁止归因为"ViT 不好" | 85.8M 参数 vs 546 张训练图——数据规模撑不起注意力的自由度，卷积的平移等变是先天优势 |
| 33 | 结论与改进方向 | content | supporting | peak | 左标题+右内容 | 三条结论 + 改进清单 | evidence | 字数约250/图0/留白18% | 禁止写成三条等宽卡片 | 小样本医学影像优先选带归纳偏置的 CNN；改进走数据、增强、混合架构三条路 |
| 34 | 07 课堂实战任务 | section | hero | transition | 全屏视觉+大标题 | L1 终端线稿 | atmosphere | 字数约70/图1/留白32% | 禁止四卡片预览 | 第七章扉页 |
| 35 | 任务书：训练与评估 | content | supporting | valley | 非对称双栏 60:40 | CodeBlock + 步骤条 | evidence | 字数约250/图1/留白16% | 禁止只给命令不给验收点 | 一条命令跑通训练，三个验收点对应理解、动手能力、分析能力 |
| 36 | 判分标准与提交物 | content | supporting | valley | 左标题+右内容 | Table(评分表) | evidence | 字数约240/图0/留白18% | 禁止评分表无权重 | 100 分拆解为结果 40 / 分析 30 / 代码 20 / 表达 10 |
| 37 | 拓展挑战 | content | supporting | peak | 上大图+下方卡片 | L1 Diagram(三条进阶路径) | anchor | 字数约230/图1/留白18% | 禁止把三条路径画成等宽卡 | 从调参、换架构到可解释性，三条进阶路径对应三档能力 |
| 38 | 总结与参考文献 | ending | hero | peak | 全屏视觉+大标题 | L1 超声底图 | atmosphere | 字数约140/图1/留白28% | 禁止放烟花金装饰 | 三句话总结 + 五条可查证引用 |

### 数据落点检查（关键数字 + 判断）

- 780 张 / 恶性仅 210 张 → 样本稀缺，必须靠迁移学习与类别加权
- 437:210:133 → 分层划分 70/15/15，测试集 118 张
- ResNet50 89.8% / macro-F1 0.890 → 最优，但参数量并非最大
- ViT-B/16 53.4% / F1 0.439 → 85.8M 参数垫底，说明数据规模决定 Transformer 成败
- 恶性召回 87.5%、特异度 96.5% → 漏诊 4 例，临床上仍需医生复核
- 早停触发于 49/50 轮 → 本配置下 50 轮是合理上限
