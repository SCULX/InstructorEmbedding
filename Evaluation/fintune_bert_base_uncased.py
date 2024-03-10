# 微调bert-base-uncased这个embedding模型
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, Trainer, TrainingArguments,BertTokenizerFast
from nlp import load_dataset
from nlp import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
from scipy.special import softmax


# 使用GPU（如果可用）  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用下载本地的bert-base-uncased模型
bert_base_uncased_path = '../Model/bert-base-uncased'

print("可以使用的device为：",device)

tokenizer = BertTokenizerFast.from_pretrained(bert_base_uncased_path)
print(f"Model name or path: {tokenizer.name_or_path}")  # ensure load this model by local model, not download on the huggingface.co


# 为了快速正确处理数据集，使用transformers包提供的函数map()时需要降级dill版本，否则报错‘module 'dill._dill' has no attribute 'PY3' ’
# 安装dill==0.3.5.1即可

# 使用quora数据集，用于判断一个sample中两个句子表达的意思是否一致，一致为True，否则为False
# 使用bert-base-uncased这个基础的text embedding模型编码一个sample中的questions，1个sample
# 中有2个question, 使用token_type_ids来区分两个句子
# 使用SFT步骤,使用这个数据集的标签进行有监督的微调,在原模型最后加上一个MLP用于二分类,训练这个分类器的参数

# Prepare Data(数据集从huggingface上下载)
train_dataset = load_dataset('quora', split='train[:3%]') # 长 12129
validation_dataset = load_dataset('quora', split='train[5%:6%]') # 长4043
test_dataset = load_dataset('quora', split='train[7%:9%]') # 长8086

def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer([examples['questions'][i]['text'][0] for i in range(len(examples['questions']))],
                           [examples['questions'][i]['text'][1] for i in range(len(examples['questions']))],
                           padding=True, truncation=True, max_length=32)

        # Map labels to IDs
        result["label"] = [(1 if l else 0) for l in examples["is_duplicate"]]
        return result

train_dataset = train_dataset.map(preprocess_function, batched=True,load_from_cache_file=False)
val_dataset = test_dataset.map(preprocess_function, batched=True,load_from_cache_file=False)
test_dataset = test_dataset.map(preprocess_function, batched=True,load_from_cache_file=False)
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

print(train_dataset[0])
# 解码input_ids得到原始文本
original_text = tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=True)
print("原始文本是：",original_text)

# Built Model(设置分类的类别为2，不输出bert隐层向量,节省内存)
# 用Bert模型的[CLS]位置的768维向量后接1个线性层,相当于nn.Linear(768,2)进行分类，但是全连接层和bert的参数均会更新
model = BertForSequenceClassification.from_pretrained(bert_base_uncased_path, num_labels=2,output_hidden_states = False)

'''
如想查看model的结构, 使用如下命令
for name, param in model.named_parameters():
    print(name)
'''

def compute_metrics(pred):
    labels = pred.label_ids
    # probs = softmax(pred.predictions, axis = 1)
    # logloss = log_loss(labels, probs)
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

training_args = TrainingArguments(
    output_dir='./results', #存储结果文件的目录
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model = "accuracy",
    weight_decay=0.01,
    warmup_steps=500,
    evaluation_strategy="steps",  # 训练过程中采用的验证策略，no/steps/apoch,不验证/每个eval_steps中执行验证/每个epoch结束时验证
    logging_strategy = "steps",
    save_strategy ='steps',  # 训练时采用的checkpoint，保存策略
    logging_steps = 100,
    seed = 2020,
    logging_dir='./logs' #存储logs的目录
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer = tokenizer
)

# 使用Transformers封装好的train函数，需要使用accelerate,所以使用 pip intstall accerate -U命令
trainer.train()

'''
transformers库中的Trainer函数非常好用,省去了写训练过程和指标计算, 包括loss计算更新等, 也会
自动保存checkpoint, 但是封装程度很高, 自定义一些是实现形式比较难, 而且源码较为复杂冗长
'''

# TODO: 完全解析transformer的Trainer类, 达到可以自定义任何想要的计算损失的函数, 各个参数的细节必须完全理解

'''
checkpoint 就是之前SFT bert-base-uncased模型后的参数, 对应于BertTokenizerFast这个模型
若需要使用在quora数据集上训练好的模型去测试结果, 则使用以下的方式加载模型
# model = BertForSequenceClassification.from_pretrained('results/checkpoint-1500', num_labels=2,output_hidden_states = False)
其中result/checkpoint-1500就是在trainer.train()过程中保存的检查点, 这个既可以用于恢复意外中断的训练, 也可以用于加载微调好的模型
若想测试在test_data上的结果, 与上述的train()类似, 使用如下代码
# def compute_metrics(pred):
#     labels = pred.label_ids
#     # probs = softmax(pred.predictions, axis = 1)
#     # logloss = log_loss(labels, probs)
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall,
#     }

# training_args = TrainingArguments(
#     output_dir='./results', #存储结果文件的目录
#     overwrite_output_dir=True,
#     num_train_epochs=5,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     learning_rate=5e-5,
#     load_best_model_at_end=True,
#     metric_for_best_model = "accuracy",
#     weight_decay=0.01,
#     warmup_steps=500,
#     evaluation_strategy="steps",  # 训练过程中采用的验证策略，no/steps/apoch,不验证/每个eval_steps中执行验证/每个epoch结束时验证
#     logging_strategy = "steps",
#     save_strategy ='steps',  # 训练时采用的checkpoint，保存策略
#     logging_steps = 100,
#     seed = 2020,
#     logging_dir='./logs' #存储logs的目录
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer = tokenizer
# )

# pre = trainer.predict(test_dataset=test_dataset)
其中定义trainer都和训练时一致,只是为了重新加载model, 然后使用trainer.predict()函数就可以一步完成评估
(注： 评估结果为metrics={'test_loss': 0.7503261566162109, 'test_accuracy': 0.8212960672767746, 'test_f1': 0.7764887857695282, 
'test_precision': 0.7341327873647265, 'test_recall': 0.8240315167432699, 'test_samples_per_second': 523.993, 'test_steps_per_second': 16.395})
'''

