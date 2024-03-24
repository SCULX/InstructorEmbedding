# 查看微调之后模型的评估效果

from mteb import MTEB
from sentence_transformers import SentenceTransformer, models
import sentence_transformers
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertConfig, BertModel


# 加载微调后模型配置
config = BertConfig.from_pretrained('results/checkpoint-1500', output_hidden_states=False)  

# 加载微调后的BERT-base模型,不包含分类头
bert_model = BertModel.from_pretrained('results/checkpoint-1500', config=config)

# 创建一个编码器，它是整个transformer模型（BERT）
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

