# linear
# nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_gpqa.log 'bash scripts/probing/deepseek_1b.sh GPQA linear'
# nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_math.log 'bash scripts/probing/deepseek_1b.sh MATH linear'
# nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_aqua.log 'bash scripts/probing/deepseek_1b.sh AQUA linear'
# mlp
nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_gpqa.log 'bash scripts/probing/deepseek_1b.sh GPQA mlp'
nlprun -q sphinx -g 1 -r 80G -c 3 -o logs/probe/deepseek_1b_math.log 'bash scripts/probing/deepseek_1b.sh MATH mlp'