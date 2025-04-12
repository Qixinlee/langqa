"""
Author: Qixinlee
Description: 基于LangChain和Ollama的本地文档问答系统
""" 

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import ollama
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser

# 常量定义
DOCUMENT_PATH = os.path.join(os.path.dirname(__file__), "tcpdump.1-4.4.0.txt")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
LLM_MODEL_NAME = "llama3.1:8b"
OLLAMA_HOST = "http://localhost:11434"

# 1. 设置 Ollama 连接信息
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
ollama.DEFAULT_BASE_URL = OLLAMA_HOST

# 2. 加载文档
def load_document(document_path: str):
    try:
        loader = TextLoader(document_path)
        documents = loader.load()
        print(f"Successfully loaded document from {document_path}")
        return documents
    except Exception as e:
        print(f"Error loading document: {e}")
        exit()

# 3. 切分文档
def split_document(documents, chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts

# 4. 加载 LLM
def load_llm(llm_model_name: str):
    try:
        llm = OllamaLLM(model=llm_model_name)
        print(f"Successfully loaded LLM model: {llm_model_name}")
        return llm
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        exit()

# 5. 创建问答链 (使用 LangChain 表达式语言)
def create_qa_chain(llm):
    try:
        template = """你是一个问答机器人。请根据以下已知信息，用简洁和专业的中文回答用户的问题。
        如果无法从中得到答案，请清晰地输出 “根据已知信息无法回答该问题”。

        已知信息:
        {context}

        问题: {question}
        答案:"""
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=template
        )

        # 使用 Runnable 序列
        qa_chain = QA_CHAIN_PROMPT | llm | StrOutputParser()

        print("Successfully created QA chain.")
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        exit()

# 6. 提问 (使用 Tenacity 实现重试机制)
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=4, max=10))
def generate_answer(qa_chain, document_content: str, query: str):
    try:
        result = qa_chain.invoke({"context": document_content, "question": query})
        print("Successfully generated answer.")
        return result
    except Exception as e:
        print(f"Error during QA: {e}")
        raise

# 主程序
if __name__ == "__main__":

    # 1. 加载文档
    documents = load_document(DOCUMENT_PATH)

    # 2. 切分文档
    texts = split_document(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. 加载 LLM (使用常量)
    llm = load_llm(LLM_MODEL_NAME)

    # 4. 将所有文本块连接成一个字符串
    document_content = "\n".join([text.page_content for text in texts])

    # 5. 创建问答链
    qa_chain = create_qa_chain(llm)

    # 6. 循环提问
    while True:
        query = input("请输入你的问题 (输入 'exit' 退出): ")
        if query.lower() == "exit":
            break

        try:
            result = generate_answer(qa_chain, document_content, query)
            print("Question (中文):", query)
            print("Answer (中文):", result)

        except Exception as e:
            print(f"Failed to generate answer: {e}")
            exit()
