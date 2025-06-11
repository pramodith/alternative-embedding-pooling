import json

import click
from mteb import MTEB
from sentence_transformers import SentenceTransformer


@click.command()
@click.option("--model", required=True, type=str, help="Model name or path to load with SentenceTransformer.")
@click.option(
    "--datasets",
    multiple=True,
    type=str,
    default=None,
    help="List of MTEB dataset names to benchmark. If not specified, will use all retrieval datasets.",
)
@click.option(
    "--output",
    type=click.Path(writable=True, dir_okay=False, resolve_path=True),
    default=None,
    help="Optional: Path to save results as JSON. Prints to stdout if not specified.",
)
def main(model, datasets, output):
    """Benchmark embedding models on MTEB retrieval tasks using SentenceTransformers."""
    print(f"Loading model: {model}")
    st_model = SentenceTransformer(model)

    # Handle datasets argument: click passes () if not provided
    tasks = list(datasets) if datasets else None

    print("Running MTEB retrieval benchmark...")
    evaluation = MTEB(tasks=tasks, task_types=["Retrieval"])
    results = evaluation.run(
        st_model, output_folder=None, eval_splits=["test"]
    )  # output_folder disables MTEB's default file output

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
