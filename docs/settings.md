# settings and overrides

- defaults live in `niels_gpt/settings.py` (single source of truth for tokenizer, data, model, training, generation, benchmark, and reproducibility toggles).
- resolved configs are produced via `resolve_settings(phase, overrides_path)` which deep-merges JSON overrides into defaults and validates invariants (special tokens must exist and encode to one id).
- training entrypoints (`python -m train.run --phase pretrain|sft|pipeline`) treat `--config` as overrides; legacy full configs are auto-adapted with a warning.
- print resolved config via `python -m train.run --phase pretrain --config <json> --print_config`.
- generation defaults (stop on `<|eot|>`, role-token bans) and benchmark grids are pulled from settings; update settings to change behavior.
- audit guardrail: `python tools/audit_config_coverage.py` fails if denylisted hyperparameters or special-token literals drift outside settings/config; enforced in `tests/test_audit_config_coverage.py`.

