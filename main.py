from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

data_path = "paper_file/"

# 加载 PDF，可以加载多篇
loaders = [
    PyPDFLoader(data_path + "attention.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
#
# print(len(docs))    # pdf的页数
# print(docs[0])      # 一页的文本作为一个列表元素

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个文本块的大小。这意味着每次切分文本时，会尽量使每个块包含 chunk_size 个字符。
    chunk_overlap=30  # 每个文本块之间的重叠部分。
)

splits = text_splitter.split_documents(docs)
#
# print(len(splits))        # 分块后的块数
# print(splits[1])          # 每一块的内容
# print(splits[0].page_content)   # 一块的具体文本

content_list = []
for chunk in splits:
    content_list.append(chunk.page_content)

# print(len(content_list))

# 获得embedding
emb_model = OllamaEmbeddings(base_url="127.0.0.1:11434", model="mxbai-embed-large:latest")

embs = emb_model.embed_documents(content_list)
embeds = np.array(embs)     # 要转为数组的形式用在faiss里
#
# print(embeds.shape)
# print(embeds)

# 使用faiss构建索引
d = 1024
index = faiss.IndexFlatL2(d)  # 构建索引，FlatL2为暴力检索，L2表示相似度度量方法为L2范数（欧氏距离）
# print(index.is_trained)     # 输出为True，表示Index不需要训练，只需要add向量即可
# print(index.ntotal)     # index中包含的向量总数
index.add(embeds)

while True:
    # 输入问题作为query
    query = input("请输入问题（最好用英文，加上问号，例如 What field is this paper in?）：")
    q_emb = np.array(emb_model.embed_documents(query))
    # print(q_emb)

    k = 5       # 检索topk个相似块
    distance, k_id = index.search(q_emb, k)

    # 将结果保存为dataframe，可以保存下来
    df = pd.DataFrame(columns=['sentence', 'distance'])
    i = 0
    for d, id in zip(distance[0], k_id[0]):
        df.loc[i] = [content_list[id], d]
        i += 1

    print("检索到的内容：")
    print(df)
    print("\n")
    # 把df处理成有格式的str
    content = []
    for i, row in df.iterrows():
        content.append(row['sentence'])

    llm = ChatOllama(base_url="127.0.0.1:11434", model="gemma2:2b")
    # 用提示模板告诉llm分析检索出来的文本
    template_string = f"""\
    Now user need to read a paper. We extracted the following content from this paper: {content}.
    This is the user's question: {query}. Please answer the user's question based on the extracted content.
    """

    # print(template_string)

    prompt_template = ChatPromptTemplate(template_string)
    prompt = prompt_template.format_messages(query=query, content=content, k=2)

    response = llm.invoke(prompt)
    print("llm的回复：")
    print(response.content)
