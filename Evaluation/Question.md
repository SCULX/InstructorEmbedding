## Question
1. 在示例任务**Banking77Classification**中出现的`accuracy_stderr`是什么，暂时没有查到这个含义，也不清楚为什么需要这个`stderr`这个指标.
2. 对比在**Colba**中运行结果和**本地**运行结果，以及**MTEB排行榜**中这个任务的具体指标，发现各有差异，依次是：
> 0.6347402 -> 0.63402 -> 0.6341

    所以这种细小差异和最终评分之间是否存在误导！

3. 关于各种任务评估指标细节难以找到解释材料，但是**leaderboard**中只是一个指标，为什么只取一个指标，取哪个指标也不明晰.
4. 微调之后的模型在Banking77Classsification这个任务上优点提升（accuracy: 0.6676  VS 0.6347）, 但是指标提升可能是我加载方式和评估的模型也许不是原始架构,可能被改变了。

***
## Solution
1. no solved yet.