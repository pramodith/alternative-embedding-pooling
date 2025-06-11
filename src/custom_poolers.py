import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from mteb_utils import get_mteb_retrieval_dataset


class CustomPooler: 
    """A class that implements a custom pooler for embedding models."""
    
    def __init__(self, model_name: str, dataset_name: str):
        """Initialize the class.

        Args:
            model_name (str): The name of the embedding model.
            
        """
        self.embedding_model: SentenceTransformer = SentenceTransformer(model_name)
        self.queries, self.corpus, self.relevant_docs = get_mteb_retrieval_dataset(dataset_name)

    def pooler(self, last_hidden_layer: torch.Tensor) -> torch.Tensor:   
        """Pooler.

        Take in the last hidden layer of an embedding model of shape [B, S, H]
        and performs a custom pooling operation to return a tensor of shape [B, H]
        that represents the embedding of the document.

        Args:
            last_hidden_layer (torch.Tensor): last hidden layer of an embedding model of shape [B, S, H].

        Returns:
            torch.Tensor: Doc embedding of shape [B, H].

        """
        pass
    
    
    def embed_documents(self, batch_size: int = 512) -> torch.Tensor:
        """Embeds the corpus of documents
        
        Args:
            batch_size (int): The batch size to use
        Returns:
            torch.Tensor of embeddings for all documents
        """
        docs = [""] * len(self.corpus)
        for key, val in self.corpus.items():
            docs[int(key)] = val
        self.embedding_store = self.embedding_model.encode(docs, batch_size=batch_size)
    
    
    def retrieve_top_k_docs(self, query:str, top_k: int = 10) -> tuple:
        """Retrieves top k documents from the corpus for a given query.

        Args:
            query (str): The query
            top_k (int, optional): The number of docs to return. Defaults to 10.

        Returns:
            List[Tuple[str, str, float]]: A list of tuples where each tuple contains the document_id,
                retrieved documents text and retrieval score.
        """
        query_emb =  self.embedding_model.encode([query], prompt_name="query")
        similarity_scores = self.embedding_model.similarity(query_emb, self.embedding_store)
        top_k_scores, top_k_docs = torch.topk(similarity_scores, top_k, dim=1)        
        retrieved_docs  = [
            (doc_id, self.relevant_docs[doc_id], score.item()) for doc_id, score in (top_k_docs, top_k_scores)
        ]
        return retrieved_docs


    def get_false_positives(self, query_id: str, retrieved_docs: list[tuple[str, float]]) -> list[str]:
        """Find the false positive docs.

        Args:
            query_id (str): Query id
            retrieved_docs (List[Tuple[str, float]]): List of top-k docs retrieved

        Returns:
            List[str]: The text of the documents that are false positives
            
        """
        false_positives = []
        relevant_docs = [doc_id for doc_id, _ in self.relevant_docs[query_id]]
        for doc_id, doc_text, _ in retrieved_docs:
            if doc_id not in relevant_docs:
                false_positives.append(doc_text)
        
        return false_positives
        

if __name__ == "__main__":
    custom_pooler = CustomPooler("Qwen/Qwen3-Embedding-0.6B")
    custom_pooler.embed_documents()
    num_queries = 1
    for query_id, query in custom_pooler.queries.items():
        print(f"Query document is:\n {query}")
        retrieved_docs = custom_pooler.retrieve_top_k_docs(query)
        false_positives = custom_pooler.get_false_positives(query_id, retrieved_docs)
        for fp in false_positives:
            print(f"False Positive Document:\n {fp}")
