# langqa
使用 LangChain 框架和 Ollama 开源大模型搭建一个问答系统，根据提供的文档内容回答用户提出的问题。

系统主要由以下几个部分组成：
*   **文档加载模块:** 负责从指定的路径加载文档。
*   **文档切分模块:** 将加载的文档按照指定的大小和重叠度进行切分，以便 LLM 处理。
*   **LLM 加载模块:** 负责加载指定的 Ollama 模型。
*   **问答链创建模块:** 使用 LangChain 表达式语言 (LCEL) 构建问答链，将用户问题和文档内容输入到 LLM 中，并解析 LLM 的输出。
*   **提问模块:**  接收用户输入的问题，调用问答链生成答案，并返回给用户。
*   **重试机制:** 使用 Tenacity 库，当提问模块出现错误时，自动进行重试，以提高系统的稳定性。


#### 1. 创建一个虚拟环境，名字为langqa
```bash
conda create -n langqa python=3.12
```

#### 2. 激活虚拟环境
```bash
conda activate langqa
```

#### 3. 安装依赖
```bash
pip install langchain langchain_community langchain-ollama ollama tenacity
```

#### 4. 部署代码

替换为你自己的文件、模型和地址
- 文件：tcpdump.1-4.4.0.txt
- 模型：llama3.1:8b
- OLLAMA ：http://localhost:11434

配置完成后使用命令行进行运行对话
```bash
python localqa.py
```
