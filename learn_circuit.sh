uv run python learn_circuit.py \
  --num_new_branches 16 \
  --num_ig_steps 20 \
  --max_new_tokens 200 \
  --layers_to_analyse "all" \
  --output_dir results/circuitviz/math_all_layers \
  --ablate_non_target_layers \
  --sentence_gap 0 
#   --no_negate_scores

uv run python visualize_circuit.py --mask_path results/circuitviz/math_all_layers/circuit_nodewise_attribution_layers_all_branches16_ig20.json --top_k_heads 6 --threshold 5e-8