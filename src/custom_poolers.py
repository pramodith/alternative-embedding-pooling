import torch
from sentence_transformers import SentenceTransformer

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
    
    def embed_documents(self, batch_size: int = 512) -> torch.Tensor
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


if __name__ == "__main__":
    custom_pooler = CustomPooler("Qwen/Qwen3-Embedding-0.6B")