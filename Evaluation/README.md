## MTEB
1. MTEB 是一个方便对比各种sentence_embedding在下游任务上的表现.
2. More information are [here](https://github.com/embeddings-benchmark/mteb).
3. If you want to use it, attention it need the package **pytrec-eval**, which is currently not support on Windows (outside of Cygwin).



## Some enviroment tips
1. **MTEB** include some package needed the C/C++ complier, so for the linux os must have the gcc, if not you should install it to ensure the **pytrec-eval** susscussfully built.
2. You can create a docker container to run the whole paogram.
3. Due to the program need the **Transformers**, so you must be sure can connect to the `huggingface.co` to download some pretrained embedding model. 