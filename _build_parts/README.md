# `_build_parts/` -- assembly sources for `train_torch.py`
#
# `train_torch.py` is a single assembled file, but it was written in pieces by a
# tool with a per-edit size limit. These are the pieces it was assembled from:
#
#     train_head.py    imports, metrics, corpus loading, augmentation
#     train_offset.py  offset_stress_probs()
#     train_score.py   score_all()
#     train_setup.py   resolve_arch() + main() setup
#     train_body.py    the training loop, summary and __main__ guard
#
# They are NOT imported at runtime -- `train_torch.py` is standalone and is the
# only file to run or maintain. They are kept so the assembly is reproducible and
# auditable, and are safe to delete if that is not wanted.
#
# `explore_part1.py` / `explore_part2.py` are the two halves merged into
# `explore_arch.py`, and `extract.ps1` sliced the original source. Same status.