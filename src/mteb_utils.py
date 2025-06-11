from typing import Any

import mteb


def get_mteb_retrieval_dataset(
    dataset_name: str = "NFCorpus", 
    split: str = "test", 
    n_samples: int = 1, 
    verbose: bool = False
) -> tuple:
    """Load and inspect an MTEB retrieval dataset.

    Print a summary and sample queries/corpus/relevant docs.
    Return the same information as a dictionary.

    Args:
        dataset_name (str): Name of the MTEB dataset. Default is 'NFCorpus' (medical).
        split (str): Which split to load ('train', 'test', etc). Default is 'test'.
        n_samples (int): Number of queries to sample. Default is 1.
        verbose (bool): A flag to indicate if a sample query and relevant docs should be
            displayed out or not.

    Returns:
        Tuple[List[str], List[str], List[str]] A tuple of the queries, corpus and relevant docs.
        
    """
    # Load the dataset
    task = mteb.get_task(dataset_name, eval_splits=[split])
    task.load_data()

    queries = task.queries[split]
    corpus = task.corpus[split]
    relevant_docs = task.relevant_docs[split]

    if verbose:
        print(f"Total corpus size is : {len(corpus)}")
        for ind, query_id in enumerate(queries):
            if ind < n_samples:
                print(f"Query is:\n {queries[query_id]}")
                rel_doc_ids_for_query = [k for k in relevant_docs[query_id]]
                print(f"Total number of relevant docs are : {len(rel_doc_ids_for_query)}")
                for rel_doc_id in rel_doc_ids_for_query:
                    print(f"Relevant docs are :\n {corpus[rel_doc_id]}")
                    print()

    return queries, corpus, relevant_docs

if __name__ == "__main__":
    result = inspect_mteb_retrieval_dataset()  # Uses defaults
