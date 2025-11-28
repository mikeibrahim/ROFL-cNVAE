#!/usr/bin/env python3
"""
Generate synthetic ROFL dataset for training the cNVAE model.
This creates optical flow data for fixation trials.
"""

import numpy as np
from pathlib import Path
import os

# Import dataset utilities
from base.dataset import generate_simulation, save_simulation


def create_rofl_dataset(
    output_dir: str = '/Users/mike/berkeley/rctn/ROFL-cNVAE/data',
    sim_type: str = 'fixate1',
    total_samples: int = 750000,
    seed: int = 0,
):
    """
    Create a synthetic ROFL dataset.
    
    Args:
        output_dir: Directory to save the dataset
        sim_type: Simulation type ('fixate0', 'fixate1', 'transl0', 'transl1', 'obj1')
        total_samples: Total number of samples to generate (in thousands, divided by 1000)
        seed: Random seed for reproducibility
    """
    
    np.random.seed(seed)
    
    # Parse simulation type
    category = sim_type[:-1]  # 'fixate', 'transl', or 'obj'
    n_obj = int(sim_type[-1])  # 0 or 1
    
    print(f"Generating ROFL dataset: {sim_type}")
    print(f"  Category: {category}")
    print(f"  Number of objects: {n_obj}")
    print(f"  Total samples: {total_samples}")
    
    # Common parameters for optical flow generation
    kwargs = {
        'dim': 17,           # 17x17 flow field
        'fov': 15.0,         # Field of view
        'res': 1.0,          # Resolution
        'z_bg': 100.0,       # Background depth
        'obj_r': 5.0,        # Object radius
        'obj_bound': 5.0,    # Object boundary
        'obj_zlim': [20.0, 80.0],  # Object depth limits
        'vlim_slf': [2.0, 10.0],   # Self motion velocity limits
        'vlim_obj': [2.0, 10.0],   # Object velocity limits
        'residual': False,    # No residual motion
    }
    
    # Acceptance criteria for filtering
    accept_n = {0: 1, 1: 1}  # Minimum objects to accept
    
    # Generate simulation
    print("\nGenerating optical flow data...")
    x, g, g_aux, attrs = generate_simulation(
        category=category,
        n_obj=n_obj,
        total=total_samples,
        kwargs=kwargs,
        accept_n=accept_n,
        min_obj_size=2,  # Minimum object size in pixels
        dtype='float32',
    )
    
    print(f"  Generated data shape: {x.shape}")
    print(f"  Generative factors shape: {g.shape}")
    print(f"  Auxiliary factors shape: {g_aux.shape}")
    
    # Normalize the optical flow data
    print("\nNormalizing data...")
    x_min = np.percentile(np.abs(x), 2)
    x_max = np.percentile(np.abs(x), 98)
    x = np.clip(x, -x_max, x_max)
    x = x / (x_max + 1e-6)
    
    print(f"  Data range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Data mean: {x.mean():.4f}, std: {x.std():.4f}")
    
    # Save the dataset
    print("\nSaving dataset...")
    save_simulation(
        save_dir=output_dir,
        x=x,
        g=g,
        g_aux=g_aux,
        attrs=attrs,
    )
    
    print(f"\n✓ Dataset saved to: {output_dir}")
    
    # Print dataset structure
    dataset_dir = Path(output_dir) / f"{sim_type}_dim-{kwargs['dim']}_n-{total_samples//1000}k"
    if dataset_dir.exists():
        print(f"\nDataset structure:")
        print(f"  {dataset_dir}/")
        print(f"    attrs.npy")
        for split in ['trn', 'vld', 'tst']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                files = list(split_dir.glob('*.npy'))
                print(f"    {split}/")
                for f in sorted(files):
                    arr = np.load(f, allow_pickle=True)
                    if hasattr(arr, 'shape'):
                        print(f"      {f.name} {arr.shape}")
                    else:
                        print(f"      {f.name}")


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    output_dir = '/Users/mike/berkeley/rctn/ROFL-cNVAE/data'
    sim_type = 'fixate1'
    n_samples = 750000
    
    if len(sys.argv) > 1:
        sim_type = sys.argv[1]
    if len(sys.argv) > 2:
        n_samples = int(sys.argv[2])
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate dataset
    create_rofl_dataset(
        output_dir=output_dir,
        sim_type=sim_type,
        total_samples=n_samples,
    )
    
    print("\n✓ Done!")
