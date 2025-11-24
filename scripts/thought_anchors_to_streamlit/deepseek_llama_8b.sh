# correct base solution
uv run python thought_anchors_to_streamlit.py \
    --tokenizer_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --thought_anchors_folder "/nlp/scr/amirzur/math-rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/correct_base_solution" \
    --output_name "deepseek_llama_8b/thought_anchors_correct"

# correct base solution with forced answer
uv run python thought_anchors_to_streamlit.py \
    --tokenizer_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --thought_anchors_folder "/nlp/scr/amirzur/math-rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/correct_base_solution_forced_answer" \
    --output_name "deepseek_llama_8b/thought_anchors_correct_forced"

# incorrect base solution
uv run python thought_anchors_to_streamlit.py \
    --tokenizer_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --thought_anchors_folder "/nlp/scr/amirzur/math-rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/incorrect_base_solution" \
    --output_name "deepseek_llama_8b/thought_anchors_incorrect"

# incorrect base solution with forced answer
uv run python thought_anchors_to_streamlit.py \
    --tokenizer_name "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --thought_anchors_folder "/nlp/scr/amirzur/math-rollouts/deepseek-r1-distill-llama-8b/temperature_0.6_top_p_0.95/incorrect_base_solution_forced_answer" \
    --output_name "deepseek_llama_8b/thought_anchors_incorrect_forced"