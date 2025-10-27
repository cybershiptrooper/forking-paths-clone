# llama
nlprun -m jagupard36 -g 1 -r 80G -c 3 -o logs/fork/llama_3b_aqua.log 'bash scripts/forking_paths/llama_3b.sh AQuA'
# nlprun -q jag -g 1 -r 80G -c 3 -o logs/fork/llama_3b_gpqa.log 'bash scripts/forking_paths/llama_3b.sh GPQA'
# nlprun -q jag -g 1 -r 80G -c 3 -o logs/fork/llama_3b_gsm8k.log 'bash scripts/forking_paths/llama_3b.sh GSM8k'
nlprun -m jagupard36 -g 1 -r 80G -c 3 -o logs/fork/llama_3b_math.log 'bash scripts/forking_paths/llama_3b.sh MATH'
# nlprun -q jag -g 1 -r 80G -c 3 -o logs/fork/llama_3b_pythonio.log 'bash scripts/forking_paths/llama_3b.sh PythonIO'
# nlprun -q jag -g 1 -r 80G -c 3 -o logs/fork/llama_3b_wildjailbreak.log 'bash scripts/forking_paths/llama_3b.sh WildJailBreak'
# nlprun -q jag -g 1 -r 80G -c 3 -o logs/fork/llama_3b_justeval.log 'bash scripts/forking_paths/llama_3b.sh Just-Eval'
# deepseek
nlprun -m jagupard36 -g 1 -r 80G -c 3 -o logs/fork/deepseek_1b_aqua.log 'bash scripts/forking_paths/deepseek_1b.sh AQuA'
nlprun -m jagupard36 -g 1 -r 80G -c 3 -o logs/fork/deepseek_1b_math.log 'bash scripts/forking_paths/deepseek_1b.sh MATH'