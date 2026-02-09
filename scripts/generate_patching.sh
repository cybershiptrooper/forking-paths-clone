example_index=${1:-0}
# echo "Visualizing attention for example $example_index"

python -m expts.visualize_attention -idx $example_index --top_k_heads 50 --top_k_sentences 500 --layer 12
python -m expts.visualize_attention -idx $example_index --top_k_heads 50 --top_k_sentences 500 --layer 16
python -m expts.visualize_attention -idx $example_index --top_k_heads 50 --top_k_sentences 500 --layer 20
python -m expts.visualize_attention -idx $example_index --top_k_heads 50 --top_k_sentences 500 --layer 8

sentences_to_ablate=(5 10 20 30)
offsets=(0 10 -10)
for offset in "${offsets[@]}"; do
    for sentences in "${sentences_to_ablate[@]}"; do
        echo "Ablating sentences for example $example_index with offset $offset and $sentences sentences to ablate"
        python -m expts.ablate_attention -idx $example_index --offset_from_convergence $offset --num_sentences_to_ablate $sentences
        python -m expts.ablate_attention -idx $example_index --offset_from_convergence $offset --num_sentences_to_ablate $sentences --random_sentences
    done
done