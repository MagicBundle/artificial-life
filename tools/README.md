# Cloud compute tools (branch claude/cloud-tools; not for merging)

Helpers for the "ALife replication (cloud)" session, which runs CPU-only compute for the 3D PRs (docs/3D.md).

- `digest_matrix.py OUT.json`: 2D determinism digest matrix on the MLX CPU device (state digests every step for
  tiny f32/f16/bf16, micro, director, immigration and seasons cases, plus in-process runner cases).
- `digest_compare.py --baseline B.json --out NEW.json`: re-runs the matrix on the checked-out code and diffs it against
  a baseline; exit 0 means bit-identical.
- `cpu_device.py`: pytest plugin forcing the MLX CPU device (`PYTHONPATH=tools pytest -p cpu_device ...`). On Linux, MLX
  is CPU-only anyway.

Copy `tools/` out of the checkout (e.g. to /tmp/tools) before checking out another branch, and run the scripts from the
checkout under test with `uv run python /tmp/tools/digest_matrix.py ...`.
