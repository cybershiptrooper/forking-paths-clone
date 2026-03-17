# Head-level granularity (matches original ig50 experiments)
# uv run python -m expts.learn_circuit --config expt_configs/ig50/head/prefix_gap0.yaml
# uv run python -m expts.learn_circuit --config expt_configs/ig50/head/prefix_gap1.yaml

# # Layer-level granularity
# uv run python -m expts.learn_circuit --config expt_configs/ig50/layer/prefix_gap0.yaml
# uv run python -m expts.learn_circuit --config expt_configs/ig50/layer/prefix_gap1.yaml

# # Pair-level granularity (sentence-wise, shared across all layers/heads)
# uv run python -m expts.learn_circuit --config expt_configs/ig50/pair/prefix_gap0.yaml
# uv run python -m expts.learn_circuit --config expt_configs/ig50/pair/prefix_gap1.yaml
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets
unset HF_CACHE_DIR
# uv run python -m expts.learn_circuit --config expts/configs/reward_test.yaml
uv run python -m expts.learn_circuit --config expts/configs/activation_patching_layer.yaml
