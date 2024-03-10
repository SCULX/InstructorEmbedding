from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Define the sentence-transformers model name
model_name = "bert-base-uncased"
# 网络原因，只能下载本地模型然后再去加载
model = SentenceTransformer('../Model/bert-base-uncased')

evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/{model_name}")  