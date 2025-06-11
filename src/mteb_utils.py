"""
Utility functions for inspecting MTEB retrieval datasets.
Requires: pip install mteb datasets
"""
from typing import Any, Dict

import mteb

def inspect_mteb_retrieval_dataset(
    dataset_name: str = "FiQA2018", split: str = "test", n_samples: int = 1
    ) -> Dict[str, Any]:
    """
    Loads and inspects an MTEB retrieval dataset.
    Prints a summary and sample queries/corpus/relevant docs.
    Returns the same information as a dictionary.

    Args:
        dataset_name (str): Name of the MTEB dataset. Default is 'FiQA2018'.
        split (str): Which split to load ('train', 'test', etc). Default is 'test'.
        n_samples (int): Number of queries to sample. Default is 1.

    Returns:
        dict: Summary and sample data from the dataset.
    """
    # Load the dataset
    task = mteb.get_task(dataset_name, eval_splits=[split])    
    task.load_data()

    queries = task.queries[split]
    corpus = task.corpus[split]
    relevant_docs = task.relevant_docs[split]

    print(f"Total corpus size is : {len(corpus)}")
    for ind, query_id in enumerate(queries):
        if ind < n_samples:
            print(f"Query is:\n {queries[query_id]}")
            rel_doc_ids_for_query = [k for k in relevant_docs[query_id]]
            for rel_doc_id in rel_doc_ids_for_query:
                print(f"Relevant docs are :\n {corpus[rel_doc_id]}")
                print()

if __name__ == "__main__":
    result = inspect_mteb_retrieval_dataset()  # Uses defaults
