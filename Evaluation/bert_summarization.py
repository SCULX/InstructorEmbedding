# 测试MTEB中的SummEval任务
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import  torch
from scipy.stats import pearsonr, spearmanr
import numpy as np
import csv


def save_as_csv(data,filename):
    with open(filename,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# 定义模型
model_name = 'bert-base-uncased'
model = SentenceTransformer(model_name)
model.eval()  # 评估模式
# 不更新sentence_embedding的参数
for param in model.parameters():
    param.requires_grad = False

datapath = 'Data/summ/en'
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

# save_as_csv(embs_machine_summaries_all,'compare/my_ma_embedding.csv')
# save_as_csv(embs_human_summaries_all,'compare/my_hu_embeddings.csv')


# -------------  这种计算方式不对 --------------------------#
# for hu_embs,ma_embs in zip(embs_human_summaries_all,embs_machine_summaries_all):
#     for ma_emb in ma_embs:
#         cos_score = torch.cosine_similarity(ma_emb,hu_embs)
#         dot_score = torch.matmul(hu_embs,ma_emb)
#         cos_scores.append(torch.max(cos_score).item())
#         dot_scores.append(torch.max(dot_score).item())


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

# 不是动态的每个sample进行最大最小归一化, 这个max/min是数据集分布中的最大最小, 即scores打分规则是0-5, 所以最小值是0, 最大值是5
# 获取的min_score, max_score 不是每个样本内的最小/最大分数, 应该是数据集中分数这个属性中的最大和最小
# min_score = min(map(min, human_scores))
# max_score = max(map(max, human_scores))  

max_score = 5
min_score = 0  

normalized_scores = [
    list(map(lambda x: (x - min_score) / (max_score - min_score), inner_list))
    for inner_list in human_scores
]

# save_as_csv(normalized_scores, 'compare/my_human_scores.csv')

human_scores = torch.tensor(normalized_scores)

# 计算最终结果
cosine_spearman_scores = []
cosine_pearson_scores = []
dot_spearman_scores = []
dot_pearson_scores = []


for i, (hu_embs, ma_embs) in enumerate(zip(embs_human_summaries_all,embs_machine_summaries_all)):
    cos_scores = []
    dot_scores = []
    for ma_emb in ma_embs:
        cos_score = torch.cosine_similarity(ma_emb,hu_embs)
        dot_score = torch.matmul(hu_embs,ma_emb)
        cos_scores.append(torch.max(cos_score).item())
        dot_scores.append(torch.max(dot_score).item())
    
    cosine_spearman_scores.append(spearmanr(human_scores[i], cos_scores))
    cosine_pearson_scores.append(pearsonr(human_scores[i], cos_scores))
    dot_spearman_scores.append(spearmanr(human_scores[i], dot_scores))
    dot_pearson_scores.append(pearsonr(human_scores[i], dot_scores))




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


# 修改后的结果为 cos_sp: 0.2915    cos_pe: 0.2915  dot_sp: 0.2589   dot_pe : 0.2608