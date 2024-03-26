## 详细说明MTEB中Banking77Classification任务的具体实现

### 1. 调用通用接口evaluation.run
(1). 在内部函数会根据具体提供的task,进入不同的处理函数\

### 2. 构造数据集
(1). 根据Banking77Classification先去内部构造一些参数,如version,hf_hub_name等\

(2). 内置的load_dataset() 去下载相应的数据,并按照要求分割,大部分都要求分割为test\

### 3. 训练
(1). 得到数据后进行参数自动解, 包括batch_size, n_samples等,这些参数似乎是MTEB库内定义,用户无法传递参数修改\

>(2). 进行**下采样**数据,即在每一个epoch下,从10003行的训练数据中随机采样616条。测试数据共3080条则完全参与每个epoch的评估\
(3). 对于这个任务底层实现是`sklearn`的`LogisticRegression`,即逻辑回归实现的,并非`nn.Linear`这种线性回归\
(4). 采样的616条训练数据, `batch_size`设置为32。经过`bert`这个模型`encode`出`sentence_embedding`, shape为(32,768),而且还转化为了numpy,可能是cpu计算更快\
(5). 然后利用 `reg_model.fit(embedding,target_label)`, `target_label`的shape为(32,1),拟合完成一次,直到616条数据完全处理完得到最终模型\
(6). 得到的最终模型`reg_model`, 使用`predict()`来进行测试数据3080条的预测拟合,返回`accuracy`和`f1`分数\


(7).重复进行上述的**(2)->(6)**共10次, 最后获得10组 {accuracy:xxx, f1:xxx}分数\

### 4. 计算指标
(1). 计算从`3. 训练`得来的10组数据,计算各自平均值和方差,于是返回结果中会看到`f1_stderr`和`accuracy_stderr`\

### 5. 结束

### Attention
1. 在这个任务数据集中的train,一共是10003行, 但是MTEB库下采样10次,每次616条, 所以 6160 < 10003, 这是一个比较值得关注的点

2. 然而对于自己实现的这个任务,采用 `nn.Linear`,并不使用任何激活函数,在全部train数据集上训练20轮, 最终的测试准确率是75%

3. 但是加上F.sigmoid()函数,训练10轮,最终准确率才45% (MTEB官方为66，47%)