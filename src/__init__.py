"""
Core library for running PCA over embedding layers across training checkpoints.

Modules are organized under:
- config: loading and merging model/family configs
- hf_utils: Hugging Face checkpoint listing and downloading
- tensors: loading embedding tensors
- pca: PCA computation utilities
- pipeline: high-level orchestration for a single model
"""

