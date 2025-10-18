# llama
nlprun -a ellipse -q sphinx -g 1 -r 120G -o logs/select_llama_3b_a.log 'bash scripts/data_selection/llama_3b.sh AQuA,GPQA,GSM8k,MATH'
nlprun -a ellipse -q sphinx -g 1 -r 120G -o logs/select_llama_3b_b.log 'bash scripts/data_selection/llama_3b.sh PythonIO,WildJailBreak,Just-Eval'
# deepseek
nlprun -a ellipse -q sphinx -g 1 -r 120G -o logs/select_deepseek_1b_a.log 'bash scripts/data_selection/deepseek_1b.sh AQuA,GPQA,GSM8k,MATH'
nlprun -a ellipse -q sphinx -g 1 -r 120G -o logs/select_deepseek_1b_b.log 'bash scripts/data_selection/deepseek_1b.sh PythonIO,WildJailBreak,Just-Eval'