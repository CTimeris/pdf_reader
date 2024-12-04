基于langchain实现的一个简单的利用llm进行pdf阅读的问答demo

需要通过ollama本地部署embedding模型和大语言模型：

embedding模型：mxbai-embed-large:latest（其他embedding模型只需替换代码OllamaEmbeddings中的model名即可）

大语言模型：gemma2:2b（其他语言模型只需替换代码ChatOllama中的model名即可）

python环境要求：langchain、faiss、numpy、pandas、pypdf

改进方向：更好的文本分块和处理策略、更好的检索策略、更好的提示模板、更好的模型
