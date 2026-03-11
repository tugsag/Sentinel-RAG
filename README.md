# Simple AI project that incorporates cybersecurity principles to protect against prompt injection and data leakage


List of tech and principles used
1. Uses rag_sample_qas_from_kis.csv from Kaggle included in this repo
2. NeMo Guardrails to check inputs and outputs for prompt injection and data leakage issues 
3. LangChain, forms the RAG backbone that cleans and stores data, and interfaces with Ollama LLM model
4. ChromaDB, local vector DB that stores the context documents to guide the LLM and provide accurate information


fly_patch.py is used to force Ollama to play nice with NeMo API