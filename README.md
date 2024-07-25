# Useful LLM Stuff

A random collection of useful/interesting LLM (or related) stuff I've encountered.

## Reddit Threads
- [GPT-4o "natively" multi-modal, what does this actually mean?](https://www.reddit.com/r/MachineLearning/comments/1crzdhd/d_gpt4o_natively_multimodal_what_does_this/)
- https://www.reddit.com/r/LocalLLaMA/comments/1cr9wvg/friendly_reminder_in_light_of_gpt4o_release/

## YouTube Videos

- [RAG with a Neo4j Knowledge Graph: How it Works and How to Set It Up](https://www.youtube.com/watch?v=ftlZ0oeXYRE)
- [How to save money with Gemini Context Caching](https://www.youtube.com/watch?v=WCw1xBREoWw)
- [The Complete Guide to Building a Custom AI Agent From Scratch for PDF Report Generation (LlamaIndex)](https://www.youtube.com/watch?v=i8ldunneSW8)
- [10 years of NLP history explained in 50 concepts | From Word2Vec, RNNs to GPT](https://www.youtube.com/watch?v=uocYQH0cWTs&t=38s)
- [Why Does Diffusion Work Better than Auto-Regression?](https://www.youtube.com/watch?v=zc5NTeJbk-k)
- [LlamaIndex Webinar: Improving RAG with Advanced Parsing + Metadata Extraction](https://www.youtube.com/watch?v=V_-WNJgTvgg)


## Medium Articles
- [Getting Started with RAG](https://medium.com/neuml/getting-started-with-rag-9a0cca75f748)

## Technical Blogs

- [Fine-tune Llama 3 with PyTorch FSDP and Q-Lora on Amazon SageMaker (Philipp Schmid)](https://www.philschmid.de/sagemaker-train-deploy-llama3)
- [Introducing Query Pipelines (LlamaIndex)](https://www.llamaindex.ai/blog/introducing-query-pipelines-025dc2bb0537)
- [Every Way To Get Structured Output From LLMs (Boundary)](https://www.boundaryml.com/blog/structured-output-from-llms)
- [Sharing new research, models, and datasets from Meta FAIR](https://ai.meta.com/blog/meta-fair-research-new-releases/)
- [Optimizing AI Inference at Character.AI](https://research.character.ai/optimizing-inference/?ref=blog.character.ai)
- [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [Primer on Multimodal LLMs](https://aman.ai/primers/ai/VLM/)
- [Diffusion Models](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
- [Implementing ‘From Local to Global’ GraphRAG with Neo4j and LangChain: Constructing the Graph](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/)
- [RouterBench](https://blog.withmartian.com/post/router-bench)
- [Thoughts on Llama 3](https://www.factorialfunds.com/blog/thoughts-on-llama-3)
- [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)


## Training/Fune-tuning

* [LoRA Guide](https://huggingface.co/docs/diffusers/main/en/training/lora)


## Development

* [Cohere Toolkit](https://docs.cohere.com/docs/cohere-toolkit) - Toolkit for building Agentic RAG applications. Includes a chat interface.

* [LlamaIndex](https://www.llamaindex.ai/) - Open-source frameowrk for building Agentic RAG applications (supports local and hosted LLMs).

* [Ollama](https://ollama.com/) - Open-source, streamlined tool for running open-source LLMs locally. Supports command line and Python. Offers a convenient container image to help with deployment. All the models they support, they quantize in-house. It can run as a server/service on your machine, so instead of actually invoking via your python code, you're sending an HTTP request to the local Ollama server running: E.G. (from the ollama docs).  Ollama is an easy solution when you want to use an API for multiple different open source LLM's. It can switch from one to another llm in seconds.

* [LangChain](https://www.langchain.com/) - Open-source framework that helps developers create applications using large language models.

* [Building a Basic Agent (Guide)](https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/)

* [Pydantic Validation](https://pydantic.dev/articles/llm-validation) - Validate/get structured output from LLMs.

* [Instructor](https://github.com/jxnl/instructor) - Getting Strucuted Output From LLMs.

* [DSPy](https://github.com/stanfordnlp/dspy) - Framework for algorithmically optimizing LM prompts and weights.

* [vllm](https://github.com/vllm-project/vllm) - A high-throughput and memory-efficient inference and serving engine.

* [LLM Quantization Library](https://github.com/Vahe1994/AQLM)

* [Build Agentic RAG with LlamaIndex](https://github.com/meta-llama/llama-recipes/tree/main/recipes/3p_integrations/llamaindex/dlai_agentic_rag)

* [GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/)



## Evaluation

* [LangSmith](https://www.langchain.com/langsmith) - Framework built on the shoulders of LangChain. It's designed to track the inner workings of LLMs and AI agents within your product. It lets you debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework and seamlessly integrates with LangChain, the go-to open source framework for building with LLMs.

* [OpenAI Evals](https://github.com/openai/evals) - Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.

* [Rag Arena](https://github.com/mendableai/rag-arena) - Tool for benchmarking/testing different retrieval methods of RAGs (mostly those supported by LangChain seems like).

* [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) - Can compare RAG pipelines/rerankers. Has many open source rankers like TART, UPR, Colbert, **Flag Reranker**, etc.

## GUIs

* [LM Studio](https://lmstudio.ai/) - Easy to use desktop app for experimenting with local and open-source Large Language Models (LLMs). You can't use multiple different LLM's on the LM Studio as a server. It's more for experimenting with different LLMs/parameters and getting it setup quickly using GPU resources.

* [Open WebUI](https://github.com/open-webui/open-webui) - Open-source local host chater interface for LLMs.

## Benchmarks/Leaderboards

* [Code Generation on Human Eval Benchmark](https://paperswithcode.com/sota/code-generation-on-humaneval) - This benchmark is a dataset that measures the code generation capabilities of large language models (LLMs). It was introduced in 2021 and is based on a dataset of 164 programming challenges.

* [LMSYS Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard) - Popular LLM Benchmark website

* [MMLU with variants](https://paperswithcode.com/dataset/mmlu) - Benchmark designed to measure knowledge acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings. This makes the benchmark more challenging and more similar to how we evaluate humans. Uses multiple-choice questions. Kind of outdated because of data leakage.

* [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

* [MTEB](https://huggingface.co/spaces/mteb/leaderboard) -  Multi-task and multi-language comparison of embedding models. 

* [Aider](https://aider.chat/docs/leaderboards/) - Aider works best with LLMs which are good at editing code, not just good at writing code.

* [Needle In A Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack?ref=research.character.ai) - A simple 'needle in a haystack' analysis to test in-context retrieval ability of long context LLMs.

## Vector Databases

* [Weaviate](https://github.com/weaviate/weaviate) -  Open source vector database that stores both objects and vectors, allowing for combining vector search with structured filtering with the fault-tolerance and scalability of a cloud-native database, all accessible through GraphQL, REST, and various language clients. [Link to managed service](https://weaviate.io/).

* [Milvus](https://milvus.io/) - Open-source, highly scalable, and blazing fast vector database.

* [Pinecone](https://www.pinecone.io/) - Low-latency managed service (serverless) vector database to retrieve relevant data for search, RAG, recommendation, detection, and other applications. 

## Utility

* [Guidance/Grammar](https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance) -  Constrain the generation of a large language model with a specified grammar (e.g. JSON format).
* [Quantization](https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization)

## Scraper

* [JinaAI](https://jina.ai/reader/)

* [ReworkdAI](https://www.reworkd.ai/)

## Rerankers

* [Cohere Reranker](https://cohere.com/rerank)

## LLM UI

* [OpenRouter](https://openrouter.ai/)

## Presentation

* [Streamlit](https://streamlit.io/) -  Free, open-source framework that allows users to build and share web apps for machine learning and data science. If you're looking to quickly create simple web applications, especially for data visualization and dashboards, Streamlit might be a better choice (instead of *Gradio*).

* [Gradio](https://www.gradio.app/) - Open-source Python package that allows you to quickly build a demo or web application for your machine learning model, API, or any arbitrary Python function. You can then share a link to your demo or web application in just a few seconds using Gradio's built-in sharing features.

## Human-Centered Artificial Intelligence (HCAI)

* [Design@Large: 40 Years of Chasing Users Down Rabbit Holes (YouTube Video)](https://www.youtube.com/watch?v=Rjx0e3kODMg)
