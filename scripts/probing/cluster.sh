# deepseek
nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_gpqa.log 'bash scripts/probing/deepseek_1b.sh GPQA'
nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_math.log 'bash scripts/probing/deepseek_1b.sh MATH'
nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_aqua.log 'bash scripts/probing/deepseek_1b.sh AQUA'