# Graph Cospectrality Analysis Research

A Python/SageMath project focused on studying cospectral graphs and their properties, with particular emphasis on the non-backtracking (Ihara) matrix.

## Overview

This project implements tools and algorithms for:
- Finding and analyzing cospectral graphs
- Computing non-backtracking matrices and their properties 
- Visualizing graph pairs/groups
- Parallel processing of large graph sets
- Testing cospectrality using various spectral invariants

## Key Features

- **Parallel Processing**: Efficient handling of large graph sets using multiprocessing and threading
- **Visualization**: Generation of PDF reports showing cospectral graphs
- **Spectral Analysis**: Computation of various spectral invariants including:
  - Ihara zeta function
  - Non-backtracking matrix spectra
  - Generalized characteristic polynomials

## Directory Structure

```
.
├── NB_Functions.ipynb          # Core functions and experiments
├── find_cospectral_pairs.py   # Algorithm implementations
├── testing_jax.py             # GPU-accelerated computations
├── graph_display.py           # Visualization utilities
├── papers/                    # Research documentation
│   └── nb.md                  # Non-backtracking matrix theory
├── 10v_cospec_groups.txt      # Data: cospectral groups
└── 10v_cospec_pairs.txt       # Data: cospectral pairs
```

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy matplotlib networkx sage-math

# Optional GPU acceleration
pip install jax jaxlib torch
```

### SageMath Setup
Ensure SageMath is installed and accessible in your Python environment:

```bash
# macOS (using Homebrew)
brew install sage
```

## Usage

### Finding Cospectral Pairs

```python
from find_cospectral_pairs import submit_task_to_process_pool_exec
from sage.graphs.graph_generators import graphs

# Generate and analyze graphs
graph_gen = graphs.nauty_geng("10 -d3")  # 10 vertices, min degree 3
submit_task_to_process_pool_exec(graph_gen)
```

### Visualizing Results

```python
from graph_display import create_graph_visualization

# Create PDF with graph visualizations
create_graph_visualization('output.pdf', graphs)
```

### Parallel Processing

```python
from testing_jax import parallel_threads_processor

# Process graphs in parallel
results = parallel_threads_processor(process_element, graph_generator)
```

## Key Components

### NB_Functions.ipynb
Core functionality including:
- Graph generation and manipulation
- Spectral analysis functions
- Cospectrality testing

### testing_jax.py
GPU-accelerated implementations for:
- Matrix operations
- Eigenvalue computation
- Parallel graph processing

### graph_display.py
Visualization tools for:
- Creating publication-quality graph plots
- Generating PDF reports
- Comparing cospectral graphs

## Theoretical Background

The project builds on spectral graph theory, particularly:
- Non-backtracking matrices and their properties
- Ihara zeta functions
- Graph cospectrality conditions

For detailed theoretical background, see `papers/nb.md`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Submit a pull request

## License

This project is for research purposes. Please cite appropriately if used in academic work.

## Acknowledgments

Based on research in spectral graph theory and non-backtracking matrices, particularly work by:
- Angel, Friedman, Hoory (2007)
- Krzakala et al. (2013)

## Contact

For questions or collaboration opportunities, please open an issue in the repository.