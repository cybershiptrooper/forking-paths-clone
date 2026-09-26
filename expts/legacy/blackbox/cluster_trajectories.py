"""Main script to process forking paths and generate clustered trajectories."""

import argparse
import json
import os
from typing import Optional

from cluster_utils.clustering import (
    AgglomerativeClusteringEngine,
    MiniBatchKMeansClusteringEngine,
)
from cluster_utils.data_loader import ForkingPathsLoader, get_forking_paths_files, load_base_data
from cluster_utils.embedding import SentenceTransformerEmbedding
from cluster_utils.trajectory import TrajectoryBuilder
from utils.utils import MODEL_METADATA


def main(
    model_name: str,
    dataset_name: str,
    start_index: int = 0,
    end_index: Optional[int] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    clustering_method: str = "minibatch",
    n_clusters: int = 100,
    distance_threshold: float = 1.0,
    linkage: str = "average",
    batch_size: int = 1024,
):
    """
    Process forking paths data and save clustered trajectories.

    Args:
        model_name: Full model name (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
        dataset_name: Dataset name (e.g., 'gpqa')
        start_index: Start index for processing
        end_index: End index for processing (None for all)
        embedding_model_name: Sentence transformer model name
        clustering_method: 'minibatch' (fast, for large datasets) or 'agglomerative'
        n_clusters: Number of clusters
        distance_threshold: Distance threshold for agglomerative clustering (only if n_clusters=None)
        linkage: Linkage method for agglomerative ('ward', 'average', 'complete', 'single')
        batch_size: Batch size for MiniBatchKMeans
    """
    # Load config
    with open("config.json") as f:
        config = json.load(f)
        forking_paths_dir = config["save_locations"]["forking_paths_folder"]
        streamlit_dir = config["save_locations"]["streamlit_folder"]

    # Get model nickname
    model_nickname = MODEL_METADATA[model_name]["nickname"]

    # Set up paths
    fp_dir = os.path.join(
        forking_paths_dir, model_nickname, dataset_name.lower() + "_old"
    )
    base_data_file = os.path.join(
        streamlit_dir, model_nickname, dataset_name.lower(), "base_data.json"
    )

    # Create output directory
    output_dir = os.path.join("data", "clustered_trajectories", model_nickname, dataset_name.lower())
    os.makedirs(output_dir, exist_ok=True)

    # Load base data
    print(f"Loading base data from {base_data_file}")
    all_base_data = load_base_data(base_data_file)

    # Get forking paths files
    fp_files = get_forking_paths_files(fp_dir)
    print(f"Found {len(fp_files)} forking paths files")

    # Determine range
    if end_index is None:
        end_index = len(fp_files)
    end_index = min(end_index, len(fp_files), len(all_base_data))

    # Initialize embedding model
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformerEmbedding(model_name=embedding_model_name)

    # Initialize clustering engine
    if clustering_method == "minibatch":
        print(
            f"Initializing MiniBatchKMeans clustering (n_clusters={n_clusters}, batch_size={batch_size})"
        )
        clustering_engine = MiniBatchKMeansClusteringEngine(
            n_clusters=n_clusters,
            batch_size=batch_size,
        )
    elif clustering_method == "agglomerative":
        print(
            f"Initializing Agglomerative clustering (n_clusters={n_clusters}, distance_threshold={distance_threshold})"
        )
        clustering_engine = AgglomerativeClusteringEngine(
            n_clusters=n_clusters if n_clusters else None,
            distance_threshold=distance_threshold,
            linkage=linkage,
        )
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")

    # Initialize trajectory builder
    trajectory_builder = TrajectoryBuilder(
        embedding_model=embedding_model,
        clustering_engine=clustering_engine,
    )

    # Process each file
    for idx in range(start_index, end_index):
        fp_file = fp_files[idx]
        base_data = all_base_data[idx]

        output_file = os.path.join(output_dir, f"{idx:02d}.json")

        # Skip if already processed
        if os.path.exists(output_file):
            print(f"Skipping {idx:02d} - already exists")
            continue

        print(f"\nProcessing {idx:02d}: {fp_file}")

        try:
            # Load data
            loader = ForkingPathsLoader(fp_file, base_data)
            print(f"  Loaded {loader.get_num_rollouts()} rollouts")
            print(f"  {len(loader.get_unique_fork_points())} unique fork points")

            # Build trajectories
            print("  Building trajectories...")
            result = trajectory_builder.build_for_prompt(loader)

            print(f"  Found {result['metadata']['num_clusters']} clusters")

            # Save result
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {output_file}")

        except Exception as e:
            print(f"  ERROR processing {idx:02d}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate clustered trajectories from forking paths"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gpqa",
        help="Dataset name",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index for processing",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for processing",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="minibatch",
        choices=["minibatch", "agglomerative"],
        help="Clustering method: 'minibatch' (fast, for large datasets) or 'agglomerative'",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=100,
        help="Number of clusters",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for agglomerative clustering (only used if clustering_method=agglomerative and n_clusters not set)",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="average",
        choices=["ward", "average", "complete", "single"],
        help="Linkage method for agglomerative clustering",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for MiniBatchKMeans clustering",
    )

    args = parser.parse_args()
    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        start_index=args.start_index,
        end_index=args.end_index,
        embedding_model_name=args.embedding_model,
        clustering_method=args.clustering_method,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
        linkage=args.linkage,
        batch_size=args.batch_size,
    )
