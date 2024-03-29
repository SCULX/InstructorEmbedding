from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import os
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentence embedding模型, 参数不更新
model_name = "bert-base-uncased"
sent_model = SentenceTransformer(model_name)
sent_model.eval()  # 评估模式
# 不更新sentence_embedding的参数
for param in sent_model.parameters():
    param.requires_grad = False


# 1. 准备数据集, 从HF加载训练集和测试集
data_path = 'Data/bank77'
dataset = load_dataset('json',data_dir=data_path)
# dataset['train'] 共10003行     dataset['test']共3080行  共77


print(len(dataset['test']['text']))

# train_dataloader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset["test"], batch_size=32, shuffle=True)


# 2. 构建模型, 只使用一个线性层
class MyBanking77ClassificationModule(nn.Module):
    def __init__(self,dim,classNum):
        super(MyBanking77ClassificationModule,self).__init__()
        self.linear = nn.Linear(dim,classNum)

    def forward(self,sentext,sentence_model):
        sentemb = sentence_model.encode(sentext, convert_to_tensor=True)
        output = self.linear(sentemb)
        # 使用CrossEntropy不能再输出使用softmax,其内部已经包含了
        # return F.sigmoid(output)
        return output
    

sentence_embedim = 768
classnum = 77
model = MyBanking77ClassificationModule(sentence_embedim,classnum)
model = model.to(device)

# 只正则化权重w,不正则化bias
weight_params = []
bias_params = []
for name,param in model.named_parameters():
    if 'weight' in name:
        weight_params.append(param)
    elif 'bias' in name:
        bias_params.append(param)

# 3. 超参数： 选什么？？
# 对比官方代码,训练10轮,官方采用LogisticRegression函数进行逻辑回归
epoches = 10

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD([
    {'params': weight_params, 'weight_decay': 1.0},
    {'params': bias_params, 'weight_decay': 0.0}
    ], lr=0.01, momentum=0.5) 


# 下采样样本数据
def getData(nsample):
    all_indices = torch.randperm(len(dataset['train'])).tolist()
    # 下采样nsample条数据的索引
    subsample_indices = all_indices[:nsample]

    # 创建Subset数据集,仅仅包含采样的样本
    subsampled_dataset = Subset(dataset['train'],subsample_indices)
    # 创建DataLoader实例
    subsampled_dataloader = DataLoader(subsampled_dataset,batch_size=32,shuffle=True)
    return subsampled_dataloader

def train(epoch,nsample):
    total_loss = 0.0
    train_dataloader = getData(nsample)
    for batch_idx, data in enumerate(tqdm(train_dataloader)):
        inputs, labels = data['text'],data['label']
        labels = labels.to(device)
        optimizer.zero_grad()
        predict = model(inputs,sent_model)
        loss = criterion(predict, labels)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

    print("training on epoch %d , the total loss is %.4f " % (epoch+1,total_loss))


def test():
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_dataloader)):
            inputs, labels = data['text'],data['label']
            labels = labels.to(device)
            predict = model(inputs,sent_model)
            _, predicted = torch.max(predict.data, dim=1)  # predicated为维度（784，1）的张量

            # 将当前batch的预测结果和真实标签累积到列表中
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    f1 = f1_score(all_labels, all_preds, average='micro')
    acc = correct / total
    print('accuracy on test set: %d %%, f1 scores is : %.4f ' % (100 * acc, f1))
    return acc, f1


if __name__ == "__main__":
    total_acc = []
    total_f1 = []
    for epoch in range(epoches):
        train(epoch,616)  # 下采样616条数据作为训练数据
        print("=========训练完成, 开始测试========")
        acc , f1 = test()
        total_acc.append(acc)
        total_f1.append(f1)
    
    acc_mean = np.mean(total_acc)
    acc_std = np.std(total_acc)

    f1_mean = np.mean(total_f1) 
    f1_std = np.std(total_f1)

    # 创建一个字典来存储这些数据
    result_dict = {
        'task': 'Bangking77Calsscification',
        'accuracy': acc_mean,
        'acc_stderr': acc_std,
        'f1': f1_mean,
        'f1_stderr': f1_std
    }

    file_path = 'results/MyMTEB/bert-base-uncased.json'
    # 创建文件所在的目录(如果不存在)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(result_dict, f, indent=4)
