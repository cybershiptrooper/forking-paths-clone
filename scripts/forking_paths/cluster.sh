# llama
nlprun -m jagupard33 -g 1 -r 80G -c 3 -o logs/fork/llama_3b_aqua.log 'bash scripts/forking_paths/llama_3b.sh AQuA'
nlprun -m jagupard33 -g 1 -r 80G -c 3 -o logs/fork/llama_3b_gpqa.log 'bash scripts/forking_paths/llama_3b.sh GPQA'
nlprun -m jagupard33 -g 1 -r 80G -c 3 -o logs/fork/llama_3b_gsm8k.log 'bash scripts/forking_paths/llama_3b.sh GSM8k'
nlprun -m jagupard38 -g 1 -r 80G -c 3 -o logs/fork/llama_3b_math.log 'bash scripts/forking_paths/llama_3b.sh MATH'
nlprun -m jagupard38 -g 1 -r 80G -c 3 -o logs/fork/llama_3b_wildjailbreak.log 'bash scripts/forking_paths/llama_3b.sh WildJailBreak'
# deepseek
# nlprun -m jagupard38 -g 1 -r 80G -c 3 -o logs/fork/deepseek_1b_aqua.log 'bash scripts/forking_paths/deepseek_1b.sh AQuA'
# nlprun -m jagupard38 -g 1 -r 80G -c 3 -o logs/fork/deepseek_1b_gpqa.log 'bash scripts/forking_paths/deepseek_1b.sh GPQA'
# nlprun -m jagupard38 -g 1 -r 80G -c 3 -o logs/fork/deepseek_1b_gsm8k.log 'bash scripts/forking_paths/deepseek_1b.sh GSM8k'
# nlprun -m jagupard33 -g 1 -r 80G -c 3 -o logs/fork/deepseek_1b_math.log 'bash scripts/forking_paths/deepseek_1b.sh MATH'
# nlprun -m jagupard33 -g 1 -r 80G -c 3 -o logs/fork/deepseek_1b_wildjailbreak.log 'bash scripts/forking_paths/deepseek_1b.sh WildJailBreak'