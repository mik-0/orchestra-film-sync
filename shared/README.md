# Shared dependencies

- [audio/](audio/README.md): reusable audio package, configurations and dependency manifests.
- [analysis/](analysis/README.md): analysis helpers imported by several RQs.
- `project_paths.py`: repository discovery, explicit script locations and cross-RQ import paths.

Entry scripts discover this directory from their own file location. This avoids relying on the working directory or installing multiple copies of the same helper.

Intermediate experiment files are expected under `shared/audio/output/` and are excluded from this release. See [INPUTS.md](../INPUTS.md) for the required files and changes to saved paths.
