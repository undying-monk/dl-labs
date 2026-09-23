# Segmentation UNet from scratch

## Scope
* Focus changes on `src/segmentation/u_net/`
* Read shared libraries when necessary.
* Do not modify other services unless explicitly required.
* Keep changes small and focused.
* Main file for execution on `notebook.ipynb`

## Structure
* `data/` — data for training and evaluation
* `datasets/` — custom datasets and data loading
* `save/` — save checkpoint and model weights
* `inference/` — inference and prediction
* `utils/` — shared helpers
* `tests/` — tests

Follow the existing project structure and patterns.

## ML Rules
* Use PyTorch patterns already established in the service.
* Keep data preprocessing and model logic separate.
* Preserve tensor shape consistency between layers.
* Do not silently change dataset labels, class mappings, or model outputs.
* Make training changes reproducible when possible.

## Testing
* Add tests for new or changed behavior.
* Test tensor shapes and important data transformations.
* Run relevant tests before finishing.
* Code Changes
* Prefer simple, readable implementations.
* Reuse existing utilities before creating new ones.
* Do not change model architecture or training behavior unless requested.
* Do not introduce new dependencies unless necessary.

## Completion

Before considering a task complete:

Ensure the implementation is consistent with the existing architecture.
Run relevant checks/tests.
Review the final diff for unintended changes.
Keep unrelated files unchanged.

Do not create commits unless explicitly requested.