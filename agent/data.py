from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import pandas as pd



def read_sample_data(path: str) -> pd.DataFrame:
    """Read sample data from a specified path string"""
    return pd.read_csv(path)


def split_data(data: pd.DataFrame, 
               topic_col: str = 'ki_topic', 
               text_col: str = 'ki_text') -> list[Document]:
    """Prepare data: combine, make metadatas, and use langchain splitter"""
    docs = []
    for index, row in data.iterrows():
        # We include the topic in the text so the embedding "knows" the subject
        combined_content = f"Topic: {row[topic_col]}\nContent: {row[text_col]}"
        
        # Store the original topic in metadata for filtering later
        doc = Document(
            page_content=combined_content,
            metadata={"topic": row[topic_col], "source_row": index}
        )
        docs.append(doc)

    # keep the 'Topic' header at the top of each chunk if possible
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_chunks = splitter.split_documents(docs)

    return final_chunks


def get_embeddings(chunked_data: list[Document], model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """Prepare embeddings from chunked data"""
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    return embedder.embed_documents([chunk.page_content for chunk in chunked_data])


def prepare_vector_db(chunked_data: list[Document],
                      embeddings: list[float],
                      write_path: str = "chroma/it_text",
                      collection_name: str = "coll") -> chromadb.api.models.Collection.Collection:
    """Make local chromadb collection"""
    client = chromadb.PersistentClient(write_path)
    collection = client.get_or_create_collection(name=collection_name)

    collection.upsert(
        ids=[str(i) for i in range(len(chunked_data))],
        embeddings=embeddings,
        documents=[chunk.page_content for chunk in chunked_data],
        metadatas=[chunk.metadata for chunk in chunked_data]
    )

    return collection


def prepare_data(data_path: str):
    data = read_sample_data(data_path)
    chunked_data = split_data(data)
    embeddings = get_embeddings(chunked_data)
    return prepare_vector_db(chunked_data, embeddings)
