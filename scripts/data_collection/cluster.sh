# llama
nlprun -m jagupard39 -g 1 -c 3 -r 80G -o logs/collect/llama_3b_a.log 'bash scripts/data_collection/llama_3b.sh AQuA,GPQA,GSM8k,MATH'
nlprun -m jagupard39 -g 1 -c 3 -r 80G -o logs/collect/llama_3b_b.log 'bash scripts/data_collection/llama_3b.sh PythonIO,WildJailBreak,Just-Eval'
# deepseek
nlprun -m jagupard33 -g 1 -c 3 -r 80G -o logs/collect/deepseek_1b_a.log 'bash scripts/data_collection/deepseek_1b.sh AQuA,GPQA,GSM8k,MATH'
nlprun -m jagupard33 -g 1 -c 3 -r 80G -o logs/collect/deepseek_1b_b.log 'bash scripts/data_collection/deepseek_1b.sh PythonIO,WildJailBreak,Just-Eval'