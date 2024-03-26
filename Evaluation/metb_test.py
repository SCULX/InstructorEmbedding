from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "bert-base-uncased"
# 网络原因，只能下载本地模型然后再去加载
model = SentenceTransformer('Model/bert-base-uncased')


# 输出模型内部的池化策略组件信息
# 下面这段代码可以查看内部池化策略，经验证，还是使用的是平均池化
# for name, module in model.named_children():
#     print(f"Layer: {name} | Module: {module}")

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")  