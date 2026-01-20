"""Main script to process forking paths and generate clustered trajectories."""

import argparse
import json
import os
from typing import Optional

from cluster_utils.clustering import AgglomerativeClusteringEngine
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
    n_clusters: Optional[int] = None,
    distance_threshold: float = 1.0,
    linkage: str = "average",
):
    """
    Process forking paths data and save clustered trajectories.

    Args:
        model_name: Full model name (e.g., 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
        dataset_name: Dataset name (e.g., 'gpqa')
        start_index: Start index for processing
        end_index: End index for processing (None for all)
        embedding_model_name: Sentence transformer model name
        n_clusters: Number of clusters (None to use distance_threshold)
        distance_threshold: Distance threshold for agglomerative clustering
        linkage: Linkage method ('ward', 'average', 'complete', 'single')
    """
    # Load config
    with open("config.json") as f:
        config = json.load(f)
        forking_paths_dir = config["save_locations"]["forking_paths_folder"]
        streamlit_dir = config["save_locations"]["streamlit_folder"]

    # Get model nickname
    model_nickname = MODEL_METADATA[model_name]["nickname"]

    # Set up paths
    fp_dir = os.path.join(forking_paths_dir, model_nickname, dataset_name.lower())
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
    print(f"Initializing clustering (n_clusters={n_clusters}, distance_threshold={distance_threshold})")
    clustering_engine = AgglomerativeClusteringEngine(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        linkage=linkage,
    )

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
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters (None to use distance_threshold)",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for agglomerative clustering",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="average",
        choices=["ward", "average", "complete", "single"],
        help="Linkage method for agglomerative clustering",
    )

    args = parser.parse_args()
    main(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        start_index=args.start_index,
        end_index=args.end_index,
        embedding_model_name=args.embedding_model,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
        linkage=args.linkage,
    )
