import random

random.seed(42)

def format_question_mmlu(example):
    question = example['question']
    choices = example['choices']
    answer = example['answer']
    answer_text = choices[answer]
    answer_letter = chr(answer + ord('A'))

    letter_choices, answer_choices = zip(*[
        (chr(a + ord('A')), option) for a, option in enumerate(choices)
        if option != "N/A" or "N/A" not in choices[:a]
    ])

    formatted_question = f"{question}\n\nChoices:\n" + '\n'.join(f"{letter}) {option}"
        for letter, option in zip(letter_choices, answer_choices)
    )
    return {
        "question": question,
        "question_with_choices": formatted_question,
        "correct_letter": answer_letter,
        "correct_answer": answer_text,
        "all_letters": letter_choices,
        "all_answers": answer_choices 
    }

def format_question_mmlu_pro(example):
    question = example['question']
    choices = example['options']
    answer = example['answer']
    answer_text = choices[ord(answer) - ord('A')]

    letter_choices, answer_choices = zip(*[
        (chr(a + ord('A')), option) for a, option in enumerate(choices)
        if option != "N/A" or "N/A" not in choices[:a]
    ])

    formatted_question = f"{question}\n\nChoices:\n" + '\n'.join(f"{letter}) {option}"
        for letter, option in zip(letter_choices, answer_choices)
    )

    return {
        "question": question,
        "question_with_choices": formatted_question,
        "correct_letter": answer,
        "correct_answer": answer_text,
        "all_letters": letter_choices,
        "all_answers": answer_choices 
    }

def format_question_aqua(example):
    question = example['question']
    choices = example['options']
    answer = example['correct']
    answer_text = choices[ord(answer) - ord('A')][2:]

    formatted_question = f"{question}\n\nChoices:\n" + '\n'.join(choices)

    return {
        "question": question,
        "question_with_choices": formatted_question,
        "correct_letter": answer,
        "correct_answer": answer_text,
        "all_letters": [c[0] for c in choices],
        "all_answers": [c[2:] for c in choices]
    }

def format_question_arc(example):
    question = example['question']
    choices = example['choices']
    answer = example['answerKey']
    if choices['label'][0] == '1':
        choices['label'] = ['A', 'B', 'C', 'D']
        answer = chr(ord(answer) - ord('1') + ord('A'))
    answer_text = choices['text'][ord(answer) - ord('A')]

    formatted_question = f"{question}\n\nChoices:\n" + '\n'.join(
        f'{l}) {c}' for l, c in zip(choices['label'], choices['text'])
    )

    return {
        "question": question,
        "question_with_choices": formatted_question,
        "correct_letter": answer,
        "correct_answer": answer_text,
        "all_letters": choices['label'],
        "all_answers": choices['text']
    }

def format_question_gpqa(example):
    question = example['Question']
    choices = [
        example['Correct Answer'],
        example['Incorrect Answer 1'],
        example['Incorrect Answer 2'],
        example['Incorrect Answer 3']
    ]
    choice_indices = list(range(4))
    random.shuffle(choice_indices)
    answer_letter = chr(choice_indices.index(0) + ord('A'))
    answer_text = example['Correct Answer']

    letter_choices, answer_choices = zip(*[
        (chr(l + ord('A')), choices[i].strip()) 
        for l, i in enumerate(choice_indices)
    ])

    formatted_question = f"{question}\n\nChoices:\n" + '\n'.join(f"{letter}) {option}"
        for letter, option in zip(letter_choices, answer_choices)
    )

    return {
        "question": question,
        "question_with_choices": formatted_question,
        "correct_letter": answer_letter,
        "correct_answer": answer_text,
        "all_letters": letter_choices,
        "all_answers": answer_choices
    }

def format_question_wildjailbreak(example):
    question = example['adversarial']
    answer = "refuse" if example['data_type'] == "adversarial_harmful" else "comply"

    return {
        "question": question,
        "question_with_choices": question,
        "correct_letter": None,
        "correct_answer": answer,
        "all_letters": None,
        "all_answers": ["refuse", "comply"]
    }

def format_question_justeval(example):
    question = example['instruction']
    answer = "refuse" if example['category'] == "safety" else "comply"

    return {
        "question": question,
        "question_with_choices": question,
        "correct_letter": None,
        "correct_answer": answer,
        "all_letters": None,
        "all_answers": ["refuse", "comply"]
    }

def format_question_aime(example):
    question = example['problem']
    answer = example["solution"]

    return {
        "question": question,
        "question_with_choices": question,
        "correct_letter": None,
        "correct_answer": answer,
        "all_letters": None,
        "all_answers": None
    }

def format_question_game_of_24(example):
    prompt_template = "Get to 24 with the following numbers: {numbers}. You can use any arithmetic operation (+, -, x, /), including parentheses, but you must use each number exactly once. Format your final expression as \\boxed{{...}}."

    question = prompt_template.format(numbers=example["Puzzles"])
    answer = example["Solution 1"]

    return {
        "question": question,
        "question_with_choices": question,
        "correct_letter": None,
        "correct_answer": answer,
        "all_letters": None,
        "all_answers": [example[f"Solution {i}"] for i in range(1, example["Num Solutions"] + 1)]
    }

def format_question_mc_evaluation(example):
    question = example['Question']
    answer_letter = example['Answer']
    answer_text = example[answer_letter]
    
    formatted_question = f"{question}\n\nChoices:\n" + '\n'.join(f"{c}) {example[c]}"
        for c in ['A', 'B', 'C', 'D']
    )

    return {
        "question": question,
        "question_with_choices": formatted_question,
        "correct_letter": answer_letter,
        "correct_answer": answer_text,
        "all_letters": ['A', 'B', 'C', 'D'],
        "all_answers": [example[l] for l in ['A', 'B', 'C', 'D']]
    }

DATASET_TO_FORMAT = {
    "MMLU": format_question_mmlu,
    "MMLU-Pro": format_question_mmlu_pro,
    "AQuA": format_question_aqua,
    "AI2-ARC": format_question_arc,
    "GPQA": format_question_gpqa,
    "WildJailBreak": format_question_wildjailbreak,
    "Just-Eval": format_question_justeval,
    "GSM8k": format_question_mc_evaluation,
    "MATH": format_question_mc_evaluation,
    "PythonIO": format_question_mc_evaluation,
    "AIME24": format_question_aime,
    "AIME25": format_question_aime,
    "Game-of-24": format_question_game_of_24
}