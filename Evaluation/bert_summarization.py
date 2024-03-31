# 测试MTEB中的SummEval任务
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import  torch
from scipy.stats import pearsonr, spearmanr
import numpy as np


# 定义模型
model_name = 'bert-base-uncased'
model = SentenceTransformer(model_name)
model.eval()  # 评估模式
# 不更新sentence_embedding的参数
for param in model.parameters():
    param.requires_grad = False

datapath = '../Data/summ/en'
# # 使用这种形式加载这个.arrow结尾的数据集, 共100行
dataset = load_dataset('arrow' ,data_dir=datapath)

ma_text_all = []
hu_text_all = []
human_scores = []
for sample in dataset['test']:
    ma_text_all.append(sample['machine_summaries'])
    hu_text_all.append(sample['human_summaries'])
    human_scores.append(sample['relevance'])

# 使用官方库中的实现方式
human_lens = [len(human_summaries) for human_summaries in hu_text_all]
machine_lens = [len(machine_summaries) for machine_summaries in ma_text_all]

ma_embs_all = model.encode(
    [text for ma_text in ma_text_all for text in ma_text],
    batch_size=32,
    convert_to_tensor=True
)

hu_embs_all = model.encode(
    [text for hu_text in hu_text_all for text in hu_text],
    batch_size = 32,
    convert_to_tensor=True
)

# 返回原始每个sample的格式
embs_human_summaries_all = np.split(hu_embs_all, np.cumsum(human_lens)[:-1])
embs_machine_summaries_all = np.split(ma_embs_all, np.cumsum(machine_lens)[:-1])

cos_scores = []
dot_scores = []

for hu_embs,ma_embs in zip(embs_human_summaries_all,embs_machine_summaries_all):
    for ma_emb in ma_embs:
        cos_score = torch.cosine_similarity(ma_emb,hu_embs)
        dot_score = torch.matmul(hu_embs,ma_emb)
        cos_scores.append(torch.max(cos_score).item())
        dot_scores.append(torch.max(dot_score).item())

# --------这种方式得到的指标非常低,分别是：-----------------
# cos_sp: 0.1933    cos_pe: 0.1933  dot_sp: 0.0386   dot_pe : 0.0496
# for hu_text,ma_text in zip(hu_text_all,ma_text_all):
#     hu_embs = model.encode(hu_text, convert_to_tensor=True)
#     ma_embs = model.encode(ma_text, convert_to_tensor=True)
#     for ma_emb in ma_embs:
#         cos_score = torch.cosine_similarity(ma_emb,hu_embs)
#         dot_score = torch.matmul(hu_embs,ma_emb)
#         cos_scores.append(torch.max(cos_score))
#         dot_scores.append(torch.max(dot_score))
# ---------------------------------------------------------

human_scores = torch.tensor(human_scores).view(-1)

# 计算最终结果
cosine_spearman_scores = []
cosine_pearson_scores = []
dot_spearman_scores = []
dot_pearson_scores = []

cos_scores = torch.tensor(cos_scores)
dot_scores = torch.tensor(dot_scores)

cosine_spearman_scores.append(spearmanr(human_scores, cos_scores))
cosine_pearson_scores.append(pearsonr(human_scores, cos_scores))
dot_spearman_scores.append(spearmanr(human_scores, dot_scores))
dot_pearson_scores.append(pearsonr(human_scores, dot_scores))


cos_sp = np.mean(cosine_spearman_scores)
cos_pe = np.mean(cosine_pearson_scores)
dot_sp = np.mean(dot_spearman_scores)
dot_pe = np.mean(dot_pearson_scores)

# 得到的结果出奇的低,和原结果对比异常低

print("cos_sp is :", cos_pe)
print("cos_pe is :", cos_pe)
print("dot_sp is :", dot_sp)
print("dot_pe is :", dot_pe)


# 换成了官方实现的方式,结果还是低到离谱
# cos_sp: 0.1933    cos_pe: 0.1933  dot_sp: 0.0386   dot_pe : 0.0496
# 和原来根本每区别