# useful-llm-tools
A random collection of useful LLM (or related) tools I've encountered.

## Tools

1. [Ollama](https://ollama.com/) - Open-source, streamlined tool for running open-source LLMs locally. Supports command line and Python. Offers a convenient container image to help with deployment. All the models they support, they quantize in-house. It can run as a server/service on your machine, so instead of actually invoking via your python code, you're sending an HTTP request to the local Ollama server running: E.G. (from the ollama docs).  Ollama is an easy solution when you want to use an API for multiple different open source LLM's. It can switch from one to another llm in seconds.
  
2. [https://lmstudio.ai/](https://lmstudio.ai/) - Tool you can use to experiment with local and open-source LLMs.  You can't use multiple different LLM's on the LM Studio as a server. It's more for experimenting with different LLMs/parameters and getting it setup quickly using GPU resources.

3. [LangChain](https://www.langchain.com/) - Open-source framework that helps developers create applications using large language models.

4. [LangSmith](https://www.langchain.com/langsmith) - Framework built on the shoulders of LangChain. It's designed to track the inner workings of LLMs and AI agents within your product. It lets you debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework and seamlessly integrates with LangChain, the go-to open source framework for building with LLMs.

5. [Weaviate](https://github.com/weaviate/weaviate) -  Open source vector database that stores both objects and vectors, allowing for combining vector search with structured filtering with the fault-tolerance and scalability of a cloud-native database, all accessible through GraphQL, REST, and various language clients. [Link to managed service](https://weaviate.io/).

6. [Gradio](https://www.gradio.app/) - Open-source Python package that allows you to quickly build a demo or web application for your machine learning model, API, or any arbitrary Python function. You can then share a link to your demo or web application in just a few seconds using Gradio's built-in sharing features.

7. [Streamlit](https://streamlit.io/) -  Free, open-source framework that allows users to build and share web apps for machine learning and data science. If you're looking to quickly create simple web applications, especially for data visualization and dashboards, Streamlit might be a better choice (instead of Gradio).

8. [OpenAI Evals](https://github.com/openai/evals) - Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.

9. [LlamaIndex](https://www.llamaindex.ai/) - Tools for building RAGs.
  
10. [Rag Arena](https://www.ragarena.com/) - Tool for benchmarking/testing different retrieval methods of RAGs (mostly those supported by LangChain seems like).
  
11. [Code Generation on Human Eval Benchmark](https://paperswithcode.com/sota/code-generation-on-humaneval) - This benchmark is a dataset that measures the code generation capabilities of large language models (LLMs). It was introduced in 2021 and is based on a dataset of 164 programming challenges. This is being provided by [Papers With Code](https://paperswithcode.com/sota).
