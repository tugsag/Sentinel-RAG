from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails import RailsConfig




def create_rag_chain(collection: 'chromadb.api.models.Collection.Collection',
                     config_path: str = 'agent/config') -> 'nemoguardrails.integrations.langchain.runnable_rails.RunnableRails':
    """Sample RAG chain with NeMo guardrails in place using local llama 3.1"""

    llm = ChatOllama(
        model="llama3.1"
    )

    template = """You are a helpful assistant. Use the following context in your response:
    {context}

    Question: {question}
    """
    
    def retrieve_and_format(query):
        retrieved_docs = collection.query(query_texts=[query])
        docs_content = "\n\n".join(retrieved_docs['documents'][0])
        return docs_content


    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retrieve_and_format, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )

    config = RailsConfig.from_path(config_path)
    guardrails = RunnableRails(config)
    chain_with_guardrails = RunnableRails(config, runnable=chain)

    return chain_with_guardrails
