uv run python learn_circuit.py \
  --num_new_branches 1 \
  --num_ig_steps 50 \
  --max_new_tokens 200 \
  --masking_algorithm nodewise_attribution_attention \
  --pair_aggregation sum \
  --layers_to_analyse "all" \
  --output_dir results/circuitviz/ig50_prefix_no_norm \
  --ablate_non_target_layers \
  --sentence_gap 0 \
  --num_random_samples 1 \
  --mask_mode prefix 
  # --no_renormalize_masked_attention

uv run python learn_circuit.py \
  --num_new_branches 1 \
  --num_ig_steps 50 \
  --max_new_tokens 200 \
  --masking_algorithm nodewise_attribution_attention \
  --pair_aggregation sum \
  --layers_to_analyse "all" \
  --output_dir results/circuitviz/ig50_prefix_gap_1_no_norm \
  --ablate_non_target_layers \
  --sentence_gap 1 \
  --num_random_samples 1 \
  --mask_mode prefix 
  # --no_renormalize_masked_attention


####################################################################

# uv run python learn_circuit.py \
#   --num_new_branches 1 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_gen \
#   --ablate_non_target_layers \
#   --sentence_gap 0 \
#   --mask_mode generation


# uv run python learn_circuit.py \
#   --num_new_branches 1 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_gen_gap_1 \
#   --ablate_non_target_layers \
#   --sentence_gap 1 \
#   --mask_mode generation

# ####################################################################

# uv run python learn_circuit.py \
#   --num_new_branches 1 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_both \
#   --ablate_non_target_layers \
#   --sentence_gap 0 \
#   --mask_mode both

# uv run python learn_circuit.py \
#   --num_new_branches 1 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_both_gap_1 \
#   --ablate_non_target_layers \
#   --sentence_gap 1 \
#   --mask_mode both



# ####################################################################
# ####################################################################
# ####################################################################

# uv run python learn_circuit.py \
#   --num_new_branches 16 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_gap_0 \
#   --ablate_non_target_layers \
#   --sentence_gap 0

# uv run python visualize_circuit.py --mask_path results/circuitviz/math_gap_0/circuit_nodewise_attribution_layers_all_branches16_ig20.json --top_k_heads 6 --threshold 5e-8


# uv run python learn_circuit.py \
#   --num_new_branches 16 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_gap_1 \
#   --ablate_non_target_layers \
#   --sentence_gap 1

# uv run python visualize_circuit.py --mask_path results/circuitviz/math_gap_1/circuit_nodewise_attribution_layers_all_branches16_ig20.json --top_k_heads 6 --threshold 5e-8


# uv run python learn_circuit.py \
#   --num_new_branches 16 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_gap_no_norm \
#   --ablate_non_target_layers \
#   --sentence_gap 0 \
#   --no_renormalize_masked_attention 
# #   --no_negate_scores

# uv run python visualize_circuit.py --mask_path results/circuitviz/math_gap_no_norm/circuit_nodewise_attribution_layers_all_branches16_ig20.json --top_k_heads 6 --threshold 5e-8

# uv run python learn_circuit.py \
#   --num_new_branches 16 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_gap_1_no_norm \
#   --ablate_non_target_layers \
#   --sentence_gap 1 \
#   --no_renormalize_masked_attention 
# #   --no_negate_scores

# uv run python visualize_circuit.py --mask_path results/circuitviz/math_gap_1_no_norm/circuit_nodewise_attribution_layers_all_branches16_ig20.json --top_k_heads 6 --threshold 5e-8

# uv run python learn_circuit.py \
#   --num_new_branches 16 \
#   --num_ig_steps 20 \
#   --max_new_tokens 200 \
#   --layers_to_analyse "all" \
#   --output_dir results/circuitviz/math_gap_1_no_negate \
#   --ablate_non_target_layers \
#   --sentence_gap 1 \
#   --no_renormalize_masked_attention \
#   --no_negate_scores

# uv run python visualize_circuit.py --mask_path results/circuitviz/math_gap_1_no_negate/circuit_nodewise_attribution_layers_all_branches16_ig20.json --top_k_heads 6 --threshold 5e-8
