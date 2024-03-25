# 查看微调之后模型的评估效果

from mteb import MTEB
from sentence_transformers import SentenceTransformer, models
from transformers import BertForSequenceClassification
from transformers import BertConfig, BertModel,BertTokenizer

# 创建一个编码器，它是整个transformer模型（BERT）
# 因为他只加载编码word的部分，所以在当前符合想要地结果
# 且我严格对比了微调之后的bertforclassification的bert参数和这个word_embedding_model中参数，参数是完全相等的,并且
# word_embedding_model中确实只有bert-base-uncased这个模型的参数结构,只是名字变化了,但确实不含bertforclassfication中多余的二分类那层的参数
word_embedding_model = models.Transformer('results/checkpoint-1500')

# 创建一个池化层，池化层使用和原始bert-base-uncased模型一样的平均池化
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

# 创建一个SentenceTransformer实例
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



evaluation = MTEB(tasks=["Banking77Classification"])
results = evaluation.run(model, output_folder=f"results/fintune/bert-base-uncased")  

