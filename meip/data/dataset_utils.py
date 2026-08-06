import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pymatgen.core.structure import Structure
import torch

if TYPE_CHECKING:
    from meip.data.dataset_jsonl import StructureJsonlDataset

def make_storage_key(mp_id: str, graph_id: str) -> str:
    """Create a storage key from mp_id and graph_id using @ separator."""
    return f"{mp_id}@{graph_id}"

def _convert_tensor_to_list(tensor: torch.Tensor) -> Any:
    """Convert a tensor to a list."""
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy().tolist()

def convert_targets_to_serializable(targets: Dict[str, Any]) -> Dict[str, Any]:
    """Convert targets to serializable format."""
    cleaned_targets = {}
    for key, value in targets.items():
        cleaned_targets[key] = _convert_tensor_to_list(value)
    return cleaned_targets

def _process_batch_for_sizes(batch_keys, jsonl_path, offset_map, graph_converter):
    """Process a batch of keys and return graph sizes."""
    batch_sizes = {}
    for mp_id, graph_id in batch_keys:
        try:
            # Read data from file
            with open(jsonl_path, "r") as f:
                f.seek(offset_map[(mp_id, graph_id)])
                line = f.readline()
                data = json.loads(line)

            # Create structure and graph
            struct = Structure.from_dict(data["structure"])
            crystal_graph = graph_converter(struct, graph_id=graph_id, mp_id=mp_id)

            # Store sizes with @ separator to avoid conflicts with existing IDs
            batch_sizes[make_storage_key(mp_id, graph_id)] = {
                "atom_graph_size": len(crystal_graph.atom_graph),
                "bond_graph_size": len(crystal_graph.bond_graph)
            }
        except Exception as e:
            print(f"Error processing sample {make_storage_key(mp_id, graph_id)}: {str(e)}")
            continue
    return batch_sizes

def compute_and_save_graph_sizes(
    dataset: "StructureJsonlDataset",
    batch_size: int = 200,
    num_workers: int = None,
    sizes_path: str = None
):
    """Compute and save graph sizes for all samples using parallel processing.

    Args:
        dataset: The dataset to compute sizes for
        batch_size (int): Number of samples to process per batch
        num_workers (int): Number of worker processes. If None, uses cpu_count()
        sizes_path (str): Path to save the sizes file. If None, uses jsonl_path + '.sizes'
    """
    # Check for duplicate keys in dataset
    print("Checking for duplicate keys in dataset...")
    seen_keys = set()
    duplicate_keys = []
    for mp_id, graph_id in dataset.keys:
        # Use the same format as stored in results
        key_string = make_storage_key(mp_id, graph_id)
        if key_string in seen_keys:
            duplicate_keys.append(key_string)
        else:
            seen_keys.add(key_string)

    if duplicate_keys:
        print(f"WARNING: Found {len(duplicate_keys)} duplicate keys in dataset:")
        for key in duplicate_keys[:10]:  # Show first 10 duplicates
            print(f"  {key}")
        if len(duplicate_keys) > 10:
            print(f"  ... and {len(duplicate_keys) - 10} more")
        print(f"Total unique keys: {len(seen_keys)}")
        print(f"Total keys (including duplicates): {len(dataset.keys)}")
    else:
        print(f"No duplicate keys found. Total unique keys: {len(seen_keys)}")

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Set sizes path
    if sizes_path is None:
        sizes_path = dataset.jsonl_path + '.sizes'

    print(f"Computing graph sizes for {len(dataset.keys)} samples")
    print(f"Using {num_workers} workers with batch size {batch_size}")

    # Create batches of keys
    batches = [dataset.keys[i:i + batch_size] for i in range(0, len(dataset.keys), batch_size)]

    # Process batches in parallel
    all_sizes = {}
    with Pool(num_workers) as pool:
        for batch_sizes in tqdm(
            pool.starmap(
                _process_batch_for_sizes,
                [(batch, dataset.jsonl_path, dataset.offset_map, dataset.graph_converter) for batch in batches]
            ),
            total=len(batches),
            desc="Computing graph sizes"
        ):
            all_sizes.update(batch_sizes)

    # Save sizes to file
    print(f"Saving {len(all_sizes)} graph sizes to {sizes_path}")
    with open(sizes_path, 'w') as f:
        json.dump(all_sizes, f)

    print("Graph size computation completed")
    print(f"Final unique results: {len(all_sizes)}")
    return all_sizes

def _process_batch_for_filter(batch_keys, jsonl_path, offset_map, graph_converter, bond_threshold, dihedral_threshold):
    """Process a batch of keys and return filtered data."""
    # set to inf if bond_threshold is None
    if bond_threshold is None:
        bond_threshold = float('inf')
    if dihedral_threshold is None:
        dihedral_threshold = float('inf')

    batch_data = []
    for mp_id, graph_id in batch_keys:
        try:
            # Read data from file
            with open(jsonl_path, "r") as f:
                f.seek(offset_map[(mp_id, graph_id)])
                line = f.readline()
                data = json.loads(line)

            # Create structure and graph
            struct = Structure.from_dict(data["structure"])
            crystal_graph = graph_converter(struct, graph_id=graph_id, mp_id=mp_id)

            # Check graph sizes
            bond_size = len(crystal_graph.bond_graph)
            dihedral_size = len(crystal_graph.dihedral_graph)

            # Skip if graph sizes exceed thresholds
            if bond_size > bond_threshold or dihedral_size > dihedral_threshold:
                continue

            # Skip if there are isolated atoms
            if crystal_graph.num_isolated_atoms > 0:
                print(f"Skipping {make_storage_key(mp_id, graph_id)} because it has {crystal_graph.num_isolated_atoms} isolated atoms")
                continue

            # Convert back to JSON line and add to filtered data
            modified_line = json.dumps(data) + '\n'
            batch_data.append(modified_line)
        except Exception as e:
            print(f"Error processing sample {make_storage_key(mp_id, graph_id)}: {str(e)}")
            continue
    return batch_data

def save_to_jsonl_with_graph_filter(
    dataset: "StructureJsonlDataset",
    output_path: str,
    bond_threshold: int = None,
    dihedral_threshold: int = None,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = None
):
    """Save filtered dataset to JSONL format based on graph size thresholds.

    Args:
        dataset: The dataset to filter
        output_path (str): Path to save the filtered JSONL file
        bond_threshold (int): Maximum allowed bond graph size
        dihedral_threshold (int): Maximum allowed dihedral graph size
        shuffle (bool): Whether to shuffle the data before processing
        seed (int): Random seed for shuffling
        num_workers (int): Number of worker processes. If None, uses cpu_count()
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize counters
    total_processed = 0
    total_filtered = 0

    # Shuffle keys if requested
    if shuffle:
        import random
        random.seed(seed)
        keys = dataset.keys.copy()  # Create a copy to avoid modifying original
        random.shuffle(keys)
    else:
        keys = dataset.keys

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Create a queue for results
    from queue import Queue
    result_queue = Queue()

    # Split keys into batches
    batch_size = max(1, len(keys) // (num_workers * 4))  # Ensure at least 4 batches per worker
    batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]

    print(f"Processing {len(batches)} batches with {num_workers} workers")

    # Process batches in parallel
    with Pool(num_workers) as pool:
        for batch_data in tqdm(
            pool.starmap(
                _process_batch_for_filter,
                [(batch, dataset.jsonl_path, dataset.offset_map, dataset.graph_converter, bond_threshold, dihedral_threshold)
                 for batch in batches]
            ),
            total=len(batches),
            desc="Processing batches"
        ):
            result_queue.put(batch_data)

    # Write filtered data to output file
    with open(output_path, "w") as outfile:
        while not result_queue.empty():
            batch_data = result_queue.get()
            for line in batch_data:
                outfile.write(line)
                total_filtered += 1
            total_processed += len(batch_data)

    # Print filtering statistics
    print("\nFiltering Results:")
    print(f"Total processed samples: {total_processed}")
    print(f"Total filtered samples: {total_filtered}")
    print(f"Removed samples: {total_processed - total_filtered}")
    print(f"Removed percentage: {(total_processed - total_filtered) / total_processed * 100:.2f}%")
    print(f"\nFiltered JSONL file saved to: {output_path}")
    if shuffle:
        print(f"Data was shuffled with seed {seed}")

def _process_batch_for_graph_analysis(batch_keys, jsonl_path, offset_map, graph_converter):
    """Process a batch of keys and return graph size statistics."""
    batch_results = {
        'mp_ids': [],
        'graph_ids': [],
        'bond_graph_sizes': [],
        'directed_bond_graph_sizes': [],
        'dihedral_graph_sizes': []
    }

    for mp_id, graph_id in batch_keys:
        try:
            # Read data from file
            with open(jsonl_path, "r") as f:
                f.seek(offset_map[(mp_id, graph_id)])
                line = f.readline()
                data = json.loads(line)

            # Create structure and graph
            struct = Structure.from_dict(data["structure"])
            crystal_graph = graph_converter(struct, graph_id=graph_id, mp_id=mp_id)

            # Collect graph sizes
            batch_results['mp_ids'].append(crystal_graph.mp_id)
            batch_results['graph_ids'].append(crystal_graph.graph_id)
            batch_results['bond_graph_sizes'].append(len(crystal_graph.bond_graph))
            batch_results['directed_bond_graph_sizes'].append(len(crystal_graph.directed_bond_graph))
            batch_results['dihedral_graph_sizes'].append(len(crystal_graph.dihedral_graph))

        except Exception as e:
            print(f"Error processing sample {make_storage_key(mp_id, graph_id)}: {str(e)}")
            continue

    return batch_results

def analyze_graph_sizes(
    dataset: "StructureJsonlDataset",
    file_basename: str = None,
    output_dir: str = "datasets/graph_statistics",
    batch_size: int = 100,
    num_workers: int = None
):
    """Analyze the distribution of bond_graph and dihedral_graph sizes with multiprocessing.

    Requires the optional dependency ``pandas``.

    Args:
        dataset: The dataset to analyze
        file_basename (str): Base name for output files. If None, uses the jsonl filename
        output_dir (str): Directory to save statistics and plots
        batch_size (int): Number of samples to process per batch
        num_workers (int): Number of worker processes. If None, uses cpu_count()
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use jsonl filename as basename if not provided
    if file_basename is None:
        file_basename = os.path.splitext(os.path.basename(dataset.jsonl_path))[0]

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Analyzing graph sizes for {len(dataset.keys)} samples")
    print(f"Using {num_workers} workers with batch size {batch_size}")

    # Create batches of keys
    num_batches = (len(dataset.keys) + batch_size - 1) // batch_size
    batches = [dataset.keys[i:i + batch_size] for i in range(0, len(dataset.keys), batch_size)]

    # Initialize result collections
    all_mp_ids = []
    all_graph_ids = []
    all_bond_graph_sizes = []
    all_directed_bond_graph_sizes = []
    all_dihedral_graph_sizes = []

    # Process batches in parallel
    with Pool(num_workers) as pool:
        batch_results = tqdm(
            pool.starmap(
                _process_batch_for_graph_analysis,
                [(batch, dataset.jsonl_path, dataset.offset_map, dataset.graph_converter) for batch in batches]
            ),
            total=num_batches,
            desc="Processing batches"
        )

        # Collect results from all batches
        for result in batch_results:
            all_mp_ids.extend(result['mp_ids'])
            all_graph_ids.extend(result['graph_ids'])
            all_bond_graph_sizes.extend(result['bond_graph_sizes'])
            all_directed_bond_graph_sizes.extend(result['directed_bond_graph_sizes'])
            all_dihedral_graph_sizes.extend(result['dihedral_graph_sizes'])

    # Create DataFrame
    df = pd.DataFrame({
        'mp_id': all_mp_ids,
        'graph_id': all_graph_ids,
        'bond_graph_size': all_bond_graph_sizes,
        'directed_bond_graph_size': all_directed_bond_graph_sizes,
        'dihedral_graph_size': all_dihedral_graph_sizes,
    })

    print(f"Successfully processed {len(df)} samples")

    # Calculate detailed statistics
    stats = df.describe()

    # Add more detailed percentiles
    percentiles = [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99]
    for col in df.columns:
        if col in ['bond_graph_size', 'directed_bond_graph_size', 'dihedral_graph_size']:
            for p in percentiles:
                stats.loc[f'{int(p*100)}%', col] = df[col].quantile(p)

    # Reorder the index to have a more logical order
    stats = stats.reindex(['count', 'mean', 'std', 'min', '25%', '50%', '75%', '85%', '90%', '95%', '99%', 'max'])

    # Save statistics
    stats.to_csv(os.path.join(output_dir, f'{file_basename}_graph_size_statistics.csv'))

    # Plot distributions
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='bond_graph_size', bins=50)
    plt.title('Bond Graph Size Distribution')
    plt.xlabel('Number of Bonds')

    plt.subplot(1, 3, 2)
    sns.histplot(data=df, x='directed_bond_graph_size', bins=50)
    plt.title('Directed Bond Graph Size Distribution')
    plt.xlabel('Number of Directed Bonds')

    plt.subplot(1, 3, 3)
    sns.histplot(data=df, x='dihedral_graph_size', bins=50)
    plt.title('Dihedral Graph Size Distribution')
    plt.xlabel('Number of Dihedrals')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_basename}_graph_size_distributions.png'))
    plt.close()  # Close the figure to free memory

    # Print summary
    print("\nGraph Size Statistics:")
    print(stats)
    print(f"\nTotal samples: {len(df)}")
    print(f"Max bond graph size: {df['bond_graph_size'].max()}")
    print(f"Max directed bond graph size: {df['directed_bond_graph_size'].max()}")
    print(f"Max dihedral graph size: {df['dihedral_graph_size'].max()}")

    # Save raw data
    df.to_csv(os.path.join(output_dir, f'{file_basename}_graph_sizes.csv'), index=False)

    return stats

def _process_batch(batch_indices: List[int], dataset: "StructureJsonlDataset") -> List[Dict]:
    """Process a batch of indices and return the processed data."""
    batch_data = []
    for idx in batch_indices:
        try:
            crystal_graph, targets = dataset[idx]
            if crystal_graph is None:
                continue
            # Convert CrystalGraph to dict with requires_grad info
            graph_dict = crystal_graph.to_dict_w_requires_grad()

            # Add targets to the dict
            data_dict = {
                "graph": graph_dict,
                "targets": convert_targets_to_serializable(targets)
            }
            batch_data.append(data_dict)
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    return batch_data

def save_to_jsonl_as_graph(
    dataset: "StructureJsonlDataset",
    output_path: str,
    batch_size: int = 200
):
    """Save the dataset to jsonl format.

    Args:
        dataset: The dataset to save
        output_path (str): Path to save the jsonl file
        batch_size (int): Number of samples to process at once
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate total number of batches
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size

    print(f"Total number of samples: {total_samples}")
    print(f"Total number of batches: {num_batches}")
    print(f"Batch size: {batch_size}")

    try:
        # Open output file
        with open(output_path, 'w') as f:
            # Process batches sequentially
            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_samples)
                batch_indices = list(range(start_idx, end_idx))

                # Process batch
                batch_data = _process_batch(batch_indices, dataset)
                if not batch_data:
                    continue

                print(f"Writing batch {batch_idx}")
                # Write each sample in the batch as a separate line
                for data in batch_data:
                    json_line = json.dumps(data)
                    f.write(json_line + '\n')

                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{num_batches} batches")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

    print(f"Dataset saved to {output_path} with {total_samples} samples")

def save_to_parquet(
    dataset: "StructureJsonlDataset",
    output_path: str,
    batch_size: int = 200,
    num_workers: int = None
):
    """Save the dataset to parquet format with multiprocessing.

    Requires the optional dependencies ``pandas`` and ``pyarrow``.

    Args:
        dataset: The dataset to save
        output_path (str): Path to save the parquet file
        batch_size (int): Number of samples to process at once
        num_workers (int): Number of worker processes. If None, uses cpu_count()
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    # Calculate total number of batches
    total_samples = len(dataset)
    num_batches = (total_samples + batch_size - 1) // batch_size

    print(f"Total number of samples: {total_samples}")
    print(f"Total number of batches: {num_batches}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")

    # Create batch indices
    batch_indices = [list(range(i, min(i + batch_size, total_samples)))
                    for i in range(0, total_samples, batch_size)]

    try:
        # Initialize parquet writer
        schema = None
        writer = None

        # Process batches in parallel
        with Pool(num_workers) as pool:
            processed_batches = 0
            for batch_data in tqdm(
                pool.starmap(_process_batch, [(indices, dataset) for indices in batch_indices]),
                total=num_batches,
                desc="Processing batches"
            ):
                if not batch_data:
                    continue

                print(f"Converting batch {processed_batches} to DataFrame")
                # Convert batch to DataFrame
                df = pd.DataFrame(batch_data)
                print(f"Converted batch {processed_batches} to DataFrame")

                # Initialize schema and writer if not done yet
                if schema is None:
                    print(f"Initializing schema and writer for batch {processed_batches}")
                    schema = pa.Table.from_pandas(df).schema
                    writer = pq.ParquetWriter(output_path, schema)
                    print(f"Initialized schema and writer for batch {processed_batches}")

                # Write batch to parquet
                print(f"Writing batch {processed_batches} to parquet")
                table = pa.Table.from_pandas(df, schema=schema)
                writer.write_table(table)
                print(f"Written batch {processed_batches} to parquet")

                processed_batches += 1
                if processed_batches % 10 == 0:
                    print(f"Processed {processed_batches}/{num_batches} batches")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise
    finally:
        # Close the writer
        if writer is not None:
            writer.close()

    print(f"Dataset saved to {output_path} with {total_samples} samples")