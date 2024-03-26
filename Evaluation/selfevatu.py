from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

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

train_dataloader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset["test"], batch_size=32, shuffle=True)


# 2. 构建模型, 只使用一个线性层
class MyBanking77ClassificationModule(nn.Module):
    def __init__(self,dim,classNum):
        super(MyBanking77ClassificationModule,self).__init__()
        self.linear = nn.Linear(dim,classNum)

    def forward(self,sentext,sentence_model):
        sentemb = sentence_model.encode(sentext, convert_to_tensor=True)
        output = self.linear(sentemb)
        return F.sigmoid(output)
    

sentence_embedim = 768
classnum = 77
model = MyBanking77ClassificationModule(sentence_embedim,classnum)
model = model.to(device)

# 3. 超参数： 选什么？？
# 对比官方代码,训练10轮,官方采用LogisticRegression函数进行逻辑回归
epoches = 10

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) 


def train(epoches):
    for epoch in range(epoches):
        total_loss = 0.0
        for batch_idx, data in enumerate(train_dataloader):
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
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            inputs, labels = data['text'],data['label']
            labels = labels.to(device)
            predict = model(inputs,sent_model)
            _, predicted = torch.max(predict.data, dim=1)  # predicated为维度（784，1）的张量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == "__main__":
    train(epoches)
    print("=========训练完成, 开始测试========")
    test()
    # 准确率才45%, 太低了

