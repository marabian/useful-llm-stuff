# useful-llm-tools
A random collection of useful LLM (or related) tools I've encountered.

## Training/Fune-tuning

* [LoRA](https://huggingface.co/docs/diffusers/main/en/training/lora)


## Development and Testing

* [Cohere Toolkit](https://docs.cohere.com/docs/cohere-toolkit) - Toolkit for building Agentic RAG applications. Includes a chat interface.

* [Open WebUI](https://github.com/open-webui/open-webui) - Open-source local host chater interface for LLMs.

* [LlamaIndex](https://www.llamaindex.ai/) - Open-source frameowrk for building Agentic RAG applications (supports local and hosted LLMs).

* [Ollama](https://ollama.com/) - Open-source, streamlined tool for running open-source LLMs locally. Supports command line and Python. Offers a convenient container image to help with deployment. All the models they support, they quantize in-house. It can run as a server/service on your machine, so instead of actually invoking via your python code, you're sending an HTTP request to the local Ollama server running: E.G. (from the ollama docs).  Ollama is an easy solution when you want to use an API for multiple different open source LLM's. It can switch from one to another llm in seconds.

* [LM Studio](https://lmstudio.ai/) - Easy to use desktop app for experimenting with local and open-source Large Language Models (LLMs). You can't use multiple different LLM's on the LM Studio as a server. It's more for experimenting with different LLMs/parameters and getting it setup quickly using GPU resources.

* [LangChain](https://www.langchain.com/) - Open-source framework that helps developers create applications using large language models.

* [LangSmith](https://www.langchain.com/langsmith) - Framework built on the shoulders of LangChain. It's designed to track the inner workings of LLMs and AI agents within your product. It lets you debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework and seamlessly integrates with LangChain, the go-to open source framework for building with LLMs.

* [OpenAI Evals](https://github.com/openai/evals) - Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.

* [Rag Arena](https://www.ragarena.com/) - Tool for benchmarking/testing different retrieval methods of RAGs (mostly those supported by LangChain seems like).


## Benchmarks

* [Code Generation on Human Eval Benchmark](https://paperswithcode.com/sota/code-generation-on-humaneval) - This benchmark is a dataset that measures the code generation capabilities of large language models (LLMs). It was introduced in 2021 and is based on a dataset of 164 programming challenges. This is being provided by [Papers With Code](https://paperswithcode.com/sota).


* [LMSYS Chatbot Arena Leaderboard
](https://chat.lmsys.org/?leaderboard) - Popular LLM Benchmark website


## Vector Databases

* [Weaviate](https://github.com/weaviate/weaviate) -  Open source vector database that stores both objects and vectors, allowing for combining vector search with structured filtering with the fault-tolerance and scalability of a cloud-native database, all accessible through GraphQL, REST, and various language clients. [Link to managed service](https://weaviate.io/).

* [Milvus](https://milvus.io/) - Open-source, highly scalable, and blazing fast vector database.

* [Pinecone](https://www.pinecone.io/) - Low-latency managed service (serverless) vector database to retrieve relevant data for search, RAG, recommendation, detection, and other applications. 

## Presentation

* [Streamlit](https://streamlit.io/) -  Free, open-source framework that allows users to build and share web apps for machine learning and data science. If you're looking to quickly create simple web applications, especially for data visualization and dashboards, Streamlit might be a better choice (instead of *Gradio*).

* [Gradio](https://www.gradio.app/) - Open-source Python package that allows you to quickly build a demo or web application for your machine learning model, API, or any arbitrary Python function. You can then share a link to your demo or web application in just a few seconds using Gradio's built-in sharing features.
