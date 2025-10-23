# llama
nlprun -q jag -g 1 -r 80G -c 3 -o logs/collect/llama_3b_a.log 'bash scripts/data_collection/llama_3b.sh AQuA,GPQA,GSM8k,MATH'
nlprun -q jag -g 1 -r 80G -c 3 -o logs/collect/llama_3b_b.log 'bash scripts/data_collection/llama_3b.sh PythonIO,WildJailBreak,Just-Eval'
# deepseek
nlprun -q jag -g 1 -r 80G -c 3 -o logs/collect/deepseek_1b_a.log 'bash scripts/data_collection/deepseek_1b.sh AQuA,GPQA,GSM8k,MATH'
nlprun -q jag -g 1 -r 80G -c 3 -o logs/collect/deepseek_1b_b.log 'bash scripts/data_collection/deepseek_1b.sh PythonIO,WildJailBreak,Just-Eval'