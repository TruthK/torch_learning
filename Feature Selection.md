# FEATURE 
**特征是某些突出性质的表现，于他而言，特征是区分事物的关键**
* 我们提取得特征中有冗余特征，对模型的性能几乎没有帮助。
* 我们提取的特征中有些可以列为噪声（或者可以称为老鼠屎），对模型的性能不仅没有帮助，还会降低模型的性能。

# FEATURE SELECTION
**feature selection 的本质就是对一个给定特征子集的优良性通过一个特定的评价标准(evaluation criterion)进行衡量．通过特征选择，原始特征集合中的冗余（redundant）特征和不相关（irrelevant）特征被除去。而有用特征得以保留。**
* 降低了模型的复杂度，节省了大量计算资源以及计算时间。
* 提高了模型的泛化能力。什么是泛化能力呢，我打个比方，一个模型是通过训 测，可是有些人长的就像噪声，对模型将会产生一定的影响，这样第一个模型的泛化能力就比第二个模型好不少，因为他不看脸，普适性更强。  

通常来说，我们要从两个方面来考虑特征选择：
* 特征是否发散：如果一个特征不发散，就是说这个特征大家都有或者非常相似，说明这个特征不需要。
* 特征和目标是否相关：与目标的相关性越高，越应该优先选择。

特征选择有三种常用的思路：  
1. 特征过滤（Filter Methods）:对各个特征按照发散性或者相关 性进行评分，对分数设定阈值或者选择靠前得分的特征。  
优点：简单，快。  
缺点:对于排序靠前的特征，如果他们相关性较强，则引入了冗 余特征，浪费了计算资源。 对于排序靠后的特征，虽然独立作 用不显著，但和其他特征想组合可能会对模型有很好的帮助， 这样就损失了有价值的特征。  
方法有：  
    * Pearson’s Correlation,：皮尔逊相关系数，是用来度量 两个变量相互关系（线性相关）的，不过更多反应两个服从 正态分布的随机变量的相关性，取值范围在 [-1,+1] 之 间。  
    * Linear Discriminant Analysis(LDA，线性判别分析)：更 像一种特征抽取方式，基本思想是将高维的特征影到最佳鉴 别矢量空间，这样就可以抽取分类信息和达到压缩特征空 间维数的效果。投影后的样本在子空间有最大可分离性。  
    * Analysis of Variance：ANOVA,方差分析，通过分析研究不 同来源的变异对总变异的贡献大小，从而确定可控因素对研 究结果影响力的大小。  
    * Chi-Square：卡方检验，就是统计样本的实际观测值与理论 推断值之间的偏离程度，实际观测值与理论推断值之间的偏 离程 度就决定卡方值的大小，卡方值越大，越不符合；卡 方值越小，偏差越小，越趋于符合。  
2. 特征筛选（Wrapper Methods）:：通过不断排除特征或者不 断选择特征，并对训练得到的模型效果进行打分，通过预测 效果评 分来决定特征的去留。    
优点：能较好的保留有价值的特征。  
缺点：会消耗巨大的计算资源和计算时间。  
方法有：
    * 前向选择法：从0开始不断向模型加能最大限度提升模型效果的特征数据用以训练，直到任何训练数据都无法提升模型表现。  
    * 后向剃除法：先用所有特征数据进行建模，再逐一丢弃贡献最低的特征来提升模型效果，直到模型效果收敛。  
    * 迭代剃除法：反复训练模型并抛弃每次循环的最优或最劣特征，然后按照抛弃的顺序给特征种类的重要性评分。  
3. 嵌入法（Embedded Methods）:有不少特征筛选和特征过滤的共性，主要的特点就是通过不同的方法去计算不同特征对于模型的贡献。  
方法：Lasso，Elastic Net，Ridge Regression，等。

