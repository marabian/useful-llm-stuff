# Useful LLM Stuff

A random collection of useful/interesting AI (mostly LLM or related) stuff I've encountered.

## Agent

[LlamaCloud Demos](https://github.com/run-llama/llamacloud-demo)

## RAG


### General

- [Getting Started with RAG](https://medium.com/neuml/getting-started-with-rag-9a0cca75f748)
- [Sentiment 2.0: From Labels to Layers](https://thegrigorian.medium.com/sentiment-2-0-from-labels-to-layers-236675e4dc84)
- [Context is King — Evaluating real-time LLM context quality with Ragas](https://emergentmethods.medium.com/context-is-king-evaluating-real-time-llm-context-quality-with-ragas-a8df8e815dc9)
- [Building a RAG pipeline with Metadata extraction](https://github.com/run-llama/llama_extract/blob/main/examples/rag/rag_metadata.ipynb)
- [RAG Stack](https://www.timescale.com/blog/the-emerging-open-source-ai-stack)

### GraphRAG

- [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- [Implementing ‘From Local to Global’ GraphRAG with Neo4j and LangChain: Constructing the Graph](https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/)
- [RAG with a Neo4j Knowledge Graph: How it Works and How to Set It Up](https://www.youtube.com/watch?v=ftlZ0oeXYRE)
- [GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/)
- [Building a Knowledge Graph From Scratch Using LLMs](https://towardsdatascience.com/building-a-knowledge-graph-from-scratch-using-llms-f6f677a17f07)

### Vector Databases

* [Weaviate](https://github.com/weaviate/weaviate) -  Open source vector database that stores both objects and vectors, allowing for combining vector search with structured filtering with the fault-tolerance and scalability of a cloud-native database, all accessible through GraphQL, REST, and various language clients. [Link to managed service](https://weaviate.io/).

* [Milvus](https://milvus.io/) - Open-source, highly scalable, and blazing fast vector database.

* [Pinecone](https://www.pinecone.io/) - Low-latency managed service (serverless) vector database to retrieve relevant data for search, RAG, recommendation, detection, and other applications. 

### Rerankers

* [Cohere Reranker](https://cohere.com/rerank)


### OSS Libraries for RAG

* [Raglite](https://github.com/superlinear-ai/raglite) - Python toolkit for Retrieval-Augmented Generation (RAG) with PostgreSQL or SQLite. [Click here](https://www.reddit.com/r/Rag/comments/1hhoy6b/raglite_a_python_package_for_the_unhobbling_of_rag/) for Reddit thread.


## Agents

- [Text-to-SQL Agent](https://www.linkedin.com/blog/engineering/ai/practical-text-to-sql-for-data-analytics)
- [Building a Basic Agent (Guide)](https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/)
- [The Complete Guide to Building a Custom AI Agent From Scratch for PDF Report Generation (LlamaIndex)](https://www.youtube.com/watch?v=i8ldunneSW8)


## OSS

- [Fine-tune Llama 3 with PyTorch FSDP and Q-Lora on Amazon SageMaker (Philipp Schmid)](https://www.philschmid.de/sagemaker-train-deploy-llama3)
- [Thoughts on Llama 3](https://www.factorialfunds.com/blog/thoughts-on-llama-3)
- [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
- [Cohere Toolkit](https://docs.cohere.com/docs/cohere-toolkit) - Toolkit for building Agentic RAG applications. Includes a chat interface.

## Document Parsing

- [markitdown](https://github.com/microsoft/markitdown) - OSS utility tool for converting various files to Markdown (e.g. PDF, PowerPoint, Word)
- [Docling](https://github.com/DS4SD/docling) - OSS Document processing tool by IBM
- [Nvidia Blueprint](https://github.com/NVIDIA-AI-Blueprints/multimodal-pdf-data-extraction) - NVIDIA AI Blueprint for multimodal PDF data extraction for enterprise RAG
- [pypdf](https://pypdf.readthedocs.io/en/stable/) - Default library for turning pdfs into text

## Structured Output from LLMs

- [Every Way To Get Structured Output From LLMs (Boundary)](https://www.boundaryml.com/blog/structured-output-from-llms)
- [Pydantic Validation](https://pydantic.dev/articles/llm-validation) - Validate/get structured output from LLMs.
- [Instructor](https://github.com/jxnl/instructor) - Getting Strucuted Output From LLMs.
- [Guidance/Grammar](https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance) -  Constrain the generation of a large language model with a specified grammar (e.g. JSON format).


## LlamaIndex

- [Introducing Query Pipelines (LlamaIndex)](https://www.llamaindex.ai/blog/introducing-query-pipelines-025dc2bb0537)
- [LlamaIndex Webinar: Improving RAG with Advanced Parsing + Metadata Extraction](https://www.youtube.com/watch?v=V_-WNJgTvgg)
- [Build Agentic RAG with LlamaIndex](https://github.com/meta-llama/llama-recipes/tree/main/recipes/3p_integrations/llamaindex/dlai_agentic_rag)
- [Building an Auto-Insurance Agentic Workflow from Scratch ](https://github.com/run-llama/llamacloud-demo/blob/main/examples/document_workflows/auto_insurance_claims/auto_insurance_claims.ipynb)

## LangChain/LangGraph

- [Building Reliable Agents with LangGraph](https://www.youtube.com/watch?v=q1gXyyLksA8)

## VLMs

- [Primer on Multimodal LLMs](https://aman.ai/primers/ai/VLM/)
- [Don\'t use LLMs for OCR](https://www.reddit.com/r/LocalLLaMA/comments/1hjfirl/comment/m36dovv/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)


## Training/Fune-tuning

* [LoRA Guide](https://huggingface.co/docs/diffusers/main/en/training/lora)

## Local LLM

* [Ollama](https://ollama.com/) - Open-source, streamlined tool for running open-source LLMs locally. Supports command line and Python. Offers a convenient container image to help with deployment. All the models they support, they quantize in-house. It can run as a server/service on your machine, so instead of actually invoking via your python code, you're sending an HTTP request to the local Ollama server running: E.G. (from the ollama docs).  Ollama is an easy solution when you want to use an API for multiple different open source LLM's. It can switch from one to another llm in seconds.

* [vllm](https://github.com/vllm-project/vllm) - A high-throughput and memory-efficient inference and serving engine.

* [Aphrodite] - A fork of vLLM meant to serve batch requests at a high speed. On the fly quantization using SmoothQuant+ (--load-in-4-bit or --load-in-smooth)

* [LLM Quantization Library](https://github.com/Vahe1994/AQLM)

* [Serving AI from the Basement](https://ahmadosman.com/blog/serving-ai-from-the-basement-part-ii/)

## Evaluation

* [RouterBench](https://blog.withmartian.com/post/router-bench)

* [LangSmith](https://www.langchain.com/langsmith) - Framework built on the shoulders of LangChain. It's designed to track the inner workings of LLMs and AI agents within your product. It lets you debug, test, evaluate, and monitor chains and intelligent agents built on any LLM framework and seamlessly integrates with LangChain, the go-to open source framework for building with LLMs.

* [OpenAI Evals](https://github.com/openai/evals) - Evals is a framework for evaluating LLMs and LLM systems, and an open-source registry of benchmarks.

* [Rag Arena](https://github.com/mendableai/rag-arena) - Tool for benchmarking/testing different retrieval methods of RAGs (mostly those supported by LangChain seems like).

* [AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) - Can compare RAG pipelines/rerankers. Has many open source rankers like TART, UPR, Colbert, **Flag Reranker**, etc.

* [promptfoo](https://www.promptfoo.dev/) - Test & secure your LLM apps (open-source)


## GUIs

* [LM Studio](https://lmstudio.ai/) - Easy to use desktop app for experimenting with local and open-source Large Language Models (LLMs). You can't use multiple different LLM's on the LM Studio as a server. It's more for experimenting with different LLMs/parameters and getting it setup quickly using GPU resources.

* [Open WebUI](https://github.com/open-webui/open-webui) - Open-source local host chater interface for LLMs.

## Benchmarks/Leaderboards

* [A benchmark for LLMs designed with test set contamination and objective evaluation in mind](https://livebench.ai/)

* [Code Generation on Human Eval Benchmark](https://paperswithcode.com/sota/code-generation-on-humaneval) - This benchmark is a dataset that measures the code generation capabilities of large language models (LLMs). It was introduced in 2021 and is based on a dataset of 164 programming challenges.

* [LMSYS Chatbot Arena Leaderboard](https://chat.lmsys.org/?leaderboard) - Popular LLM Benchmark website

* [MMLU with variants](https://paperswithcode.com/dataset/mmlu) - Benchmark designed to measure knowledge acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings. This makes the benchmark more challenging and more similar to how we evaluate humans. Uses multiple-choice questions. Kind of outdated because of data leakage.

* [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

* [MTEB](https://huggingface.co/spaces/mteb/leaderboard) -  Multi-task and multi-language comparison of embedding models. 

* [Aider](https://aider.chat/docs/leaderboards/) - Aider works best with LLMs which are good at editing code, not just good at writing code.

* [Needle In A Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack?ref=research.character.ai) - A simple 'needle in a haystack' analysis to test in-context retrieval ability of long context LLMs.

* [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)


## Prompt Optimization

- [LLM Prompting Mental Framework by IndyDevDan](https://www.youtube.com/watch?v=pytSbBRoFw8) - Categorizing prompts into 6 different types based on use-case
- [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) - Library for auto-optimizing LLM prompts
- [DSPy](https://github.com/stanfordnlp/dspy) - Framework for algorithmically optimizing LM prompts and weights.
- [Article about prompting by Francois Chollet](https://fchollet.substack.com/p/how-i-think-about-llm-prompt-engineering)

## LLM Monitoring

- [promptfoo](https://www.promptfoo.dev/)  - OSS tool for testing & securing your LLM apps

## Tooling

- [e2b](https://e2b.dev/docs) - OSS version of Claude's artifacts (run Python/JS code in sandbox and render output)

## Web Scrapers/Browser Autmation

* [JinaAI](https://jina.ai/reader/)

* [ReworkdAI](https://www.reworkd.ai/) - API for scraping webpages into LLM ingestable formats.

* [steel-broswer](https://github.com/steel-dev/steel-browser) - The open-source browser API for AI agents & apps.

* [browser-use](https://github.com/browser-use/browser-use) - Popular OSS library for using web browser with LLMs

## LLM GUIs

* [OpenRouter](https://openrouter.ai/)


## Human-Centered Artificial Intelligence (HCAI)

* [Design@Large: 40 Years of Chasing Users Down Rabbit Holes (YouTube Video)](https://www.youtube.com/watch?v=Rjx0e3kODMg)


## Theory

* [What is Perplexity?](https://www.comet.com/site/blog/perplexity-for-llm-evaluation/)
* [Quantization](https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization)

## Tutorials

### Gemini/Vertex AI

- [Multimodal Live API demo: GenList](https://www.youtube.com/watch?v=gbObKqfqdlM)
- [Gemini 2.0 Get Started Cookbook](https://github.com/google-gemini/cookbook/blob/main/gemini-2/get_started.ipynb)

## Misc

### Technical Blogs

- [Sharing new research, models, and datasets from Meta FAIR](https://ai.meta.com/blog/meta-fair-research-new-releases/)
- [Optimizing AI Inference at Character.AI](https://research.character.ai/optimizing-inference/?ref=blog.character.ai)
- [Diffusion Models](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)


### Reddit Threads
- [GPT-4o "natively" multi-modal, what does this actually mean?](https://www.reddit.com/r/MachineLearning/comments/1crzdhd/d_gpt4o_natively_multimodal_what_does_this/)
- https://www.reddit.com/r/LocalLLaMA/comments/1cr9wvg/friendly_reminder_in_light_of_gpt4o_release/

### LinkedIn

- [Top LLM Papers of the Week](https://www.linkedin.com/pulse/top-rag-papers-week-december-1-2024-kalyan-ks-aaecc/)

### YouTube Videos

- [How to save money with Gemini Context Caching](https://www.youtube.com/watch?v=WCw1xBREoWw)
- [10 years of NLP history explained in 50 concepts | From Word2Vec, RNNs to GPT](https://www.youtube.com/watch?v=uocYQH0cWTs&t=38s)
- [Why Does Diffusion Work Better than Auto-Regression?](https://www.youtube.com/watch?v=zc5NTeJbk-k)
- [ICML 2024 Tutorial: Physics of Language Models](https://www.youtube.com/watch?v=yBL7J0kgldU)
