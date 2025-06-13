
import click
import numpy as np
import string
import torch
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer
from typing import Optional

from mteb_utils import get_mteb_retrieval_dataset
from mteb import MTEB

class SinkTokenPooling(models.Pooling):
    """Pool the sink tokens."""

    def __init__(
        self,
        n_sink_tokens: int,
        tokenizer: AutoTokenizer,
        word_embedding_dimension: int,
        pooling_mode: str | None = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
        include_prompt: bool = True,
    ):
        """Construct.

        Args:
            n_sink_tokens: The first n_sink_tokens will be used in the pooling process.
            word_embedding_dimension: Dimensions for the word embeddings
            pooling_mode: Either "cls", "lasttoken", "max", "mean",
                "mean_sqrt_len_tokens", or "weightedmean". If set,
                overwrites the other pooling_mode_* settings
            pooling_mode_cls_token: Use the first token (CLS token) as text
                representations
            pooling_mode_max_tokens: Use max in each dimension over all
                tokens.
            pooling_mode_mean_tokens: Perform mean-pooling
            pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but
                divide by sqrt(input_length).
            pooling_mode_weightedmean_tokens: Perform (position) weighted
                mean pooling. See `SGPT: GPT Sentence Embeddings for
                Semantic Search <https://arxiv.org/abs/2202.08904>`_.
            pooling_mode_lasttoken: Perform last token pooling. See `SGPT:
                GPT Sentence Embeddings for Semantic Search
                <https://arxiv.org/abs/2202.08904>`_ and `Text and Code
                Embeddings by Contrastive Pre-Training
                <https://arxiv.org/abs/2201.10005>`_.
            include_prompt: If set to false, the prompt tokens are not
                included in the pooling. This is useful for reproducing
                work that does not include the prompt tokens in the pooling
                like INSTRUCTOR, but otherwise not recommended.

        """
        super().__init__(
            word_embedding_dimension=word_embedding_dimension,  # Replace `None` with the actual dimension if known
            pooling_mode=pooling_mode,
            pooling_mode_cls_token=pooling_mode_cls_token,
            pooling_mode_max_tokens=pooling_mode_max_tokens,
            pooling_mode_mean_tokens=pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens=pooling_mode_mean_sqrt_len_tokens,
            pooling_mode_weightedmean_tokens=pooling_mode_weightedmean_tokens,
            pooling_mode_lasttoken=pooling_mode_lasttoken,
            include_prompt=include_prompt,
        )
        self.tokenizer = tokenizer
        self.n_sink_tokens = n_sink_tokens

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_embeddings = features["token_embeddings"]
        punct_mask = []
        special_tokens = self.tokenizer.all_special_tokens
        if self.tokenizer.pad_token:
            special_tokens.remove(self.tokenizer.pad_token)
        if self.tokenizer.mask_token:
            special_tokens.remove(self.tokenizer.mask_token)
        for input_seq in features["input_ids"]:
            punct_mask.append([])
            tokens = self.tokenizer.convert_ids_to_tokens(input_seq.tolist())
            for token in tokens:
                if token in string.punctuation or token in self.tokenizer.all_special_tokens:
                    punct_mask[-1].append(1)
                else:
                    punct_mask[-1].append(0)
        
        punct_mask = torch.Tensor(punct_mask)
                
        attention_mask = (
            features["attention_mask"]
            if "attention_mask" in features
            else torch.ones(token_embeddings.shape[:-1], device=token_embeddings.device, dtype=torch.int64)
        )
        if not self.include_prompt and "prompt_length" in features:
            prompt_length = features["prompt_length"]
            # prompt_length is either:
            # * an int (in inference)
            # * a tensor of shape (bs), all the same value (in training with an IterableDataset)
            # * a tensor of shape (1) (in training with a Dataset)
            # We turn all into an int
            if isinstance(prompt_length, torch.Tensor):
                prompt_length = prompt_length[0].item()

            attention_mask[:, :prompt_length] = 0

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            # find the last index where the attention mask is 1
            last_token_index = torch.argmax(torch.cumsum(attention_mask, dim=1), dim=1)
            attention_mask[:, self.n_sink_tokens:] = 0
            attention_mask = attention_mask.scatter(1, last_token_index.unsqueeze(1), 1)
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        
        output_vector = torch.cat(output_vectors, 1)
        features["sentence_embedding"] = output_vector
        return features


class CustomQwenEmbeddingModel:
    """A class that implements a custom pooler for embedding models."""

    def __init__(
        self, 
        model_name: str, 
        dataset_name: str, 
        is_baseline: bool = False, 
        n_sink_tokens: int = 16,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        """Initialize the class.

        Args:
            model_name (str): The name of the embedding model.
            dataset_name (str): The name of the dataset to download
            is_baseline (bool): Whether the default pooling mechanism should be used or not
            n_sink_tokens (int): Number of sink tokens to use
            tokenizer (AutoTokenizer): The tokenizer corresponding to the embedding model.

        """
        
        if not is_baseline:
            self.transformer = models.Transformer(model_name)
            self.sink_token_pooler = SinkTokenPooling(
                n_sink_tokens, 
                tokenizer,
                1024, 
                pooling_mode=None, 
                pooling_mode_lasttoken=True
            )
            self.normalize = models.Normalize()
            self.embedding_model: SentenceTransformer = SentenceTransformer(
                modules=[self.transformer, self.sink_token_pooler, self.normalize]
            )
            self.embedding_model.model_card_data.model_name = f"{model_name}-sink-pooler"
        else:
            self.embedding_model: SentenceTransformer = SentenceTransformer(model_name)
        self.dataset_name = dataset_name
        self.queries, self.corpus, self.relevant_docs = get_mteb_retrieval_dataset(dataset_name)

    def load_embedded_documents(self, load_path: str):
        """Load the saved embeddings.

        Args:
            load_path (str): The path to load from

        """
        self.embedding_store = np.load(load_path)
        self.embedding_store = torch.Tensor(self.embedding_store)
        if torch.cuda.is_available():
            self.embedding_store = self.embedding_store.to("cuda")

    def embed_documents(self, save_path: str, batch_size: int = 512):
        """Embed the corpus of documents and save embeddings as a numpy file.

        Args:
            save_path (str): Path to save the numpy file.
            batch_size (int): The batch size to use.

        """
        self.embedding_store = self.embedding_model.encode(
            list(self.corpus.values()),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize=True,
        )
        np.save(save_path, self.embedding_store)
        return self.embedding_store


    def retrieve_top_k_docs(self, query: str, top_k: int = 10) -> tuple:
        """Retrieve top k documents from the corpus for a given query.

        Args:
            query (str): The query
            top_k (int, optional): The number of docs to return. Defaults to 10.

        Returns:
            List[Tuple[str, str, float]]: A list of tuples where each tuple contains the document_id,
                retrieved documents text and retrieval score.

        """
        query_emb = self.embedding_model.encode([query])
        similarity_scores = self.embedding_model.similarity(query_emb, self.embedding_store)
        top_k_scores, top_k_docs = torch.topk(similarity_scores, top_k, dim=1)
        top_k_scores = top_k_scores[0]
        top_k_docs = top_k_docs[0]
        corpus_doc_ids = list(self.corpus.keys())
        top_k_docs = [corpus_doc_ids[i] for i in top_k_docs]
        retrieved_docs = [
            (doc_id, self.corpus[doc_id], score.item()) for doc_id, score in zip(top_k_docs, top_k_scores, strict=False)
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
        relevant_docs = [doc_id for doc_id, _ in self.relevant_docs[query_id].items()]
        for doc_id, doc_text, _ in retrieved_docs[: len(relevant_docs)]:
            if doc_id not in relevant_docs:
                false_positives.append(doc_text)

        return false_positives
    
    def benchmark_model(self, batch_size: int = 16):
        """Benchmarks the custom model on the dataset using MTEB.
        
        Args:
            batch_size (int): The batch size to use.
        
        Returns:
            List[Dict]: The results of the benchmark.
        """
        # Create a SentenceTransformer-like wrapper for the custom model if needed
        # Here, self.embedding_model is already a SentenceTransformer instance
        results = MTEB(tasks=[self.dataset_name]).run(
            self.embedding_model, verbosity=2, batch_size=batch_size, prompt_name="query"
        )
        for result in results:
            print(result.model_dump())
        return results


@click.command()
@click.option("--model-name", default="sentence-transformers/all-minilm-l6-v2", help="The name of the embedding model.")
@click.option("--dataset-name", default="ArguAna", help="The name of the dataset to use.")
@click.option("--batch-size", default=16, show_default=True, help="Batch size for embedding.")
@click.option("--num-queries", default=1, show_default=True, help="Number of queries to process.")
@click.option(
    "--do-benchmark", 
    is_flag=True, 
    default=True, 
    help="Whether to benchmark the model on the dataset"
)
@click.option(
    "--is-baseline", 
    is_flag=True, 
    default=False, 
    help="Whether to use the default pooler or not."
)
@click.option("--n-sink-tokens", default=16, show_default=True, help="Number of sink tokens to use.")
def main(model_name, dataset_name, batch_size, num_queries, do_benchmark, is_baseline, n_sink_tokens):
    print(f"Args are :{model_name}, {dataset_name}, {batch_size}, {num_queries}, {do_benchmark}, {is_baseline}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    custom_pooler = CustomQwenEmbeddingModel(model_name, dataset_name, is_baseline, n_sink_tokens, tokenizer)
    save_path = f"./data/{model_name.split('/')[1]}_{dataset_name}.npy"
    if not do_benchmark:
        custom_pooler.embed_documents(save_path, batch_size=batch_size)
        for ind, (query_id, query) in enumerate(custom_pooler.queries.items()):
            print(f"Query document is:\n {query}")
            retrieved_docs = custom_pooler.retrieve_top_k_docs(query)
            false_positives = custom_pooler.get_false_positives(query_id, retrieved_docs)
            for fp in false_positives:
                print(f"False Positive Document:\n {fp}\n")
            if ind + 1 == num_queries:
                break
    else:
        print(f"Benchmarking to start.")
        custom_pooler.benchmark_model(batch_size=batch_size)


if __name__ == "__main__":
    main()
