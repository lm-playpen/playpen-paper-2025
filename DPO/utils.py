games = ['wordle', 'wordle_withclue', 'wordle_withcritic', 'imagegame', 'referencegame', 'taboo', 'privateshared']

top_10_models_old_clembench = [
    'gpt-4-0613-t0.0--gpt-4-0613-t0.0',
    'claude-v1.3-t0.0--claude-v1.3-t0.0',
    'gpt-4-1106-preview-t0.0--gpt-4-1106-preview-t0.0',
    'gpt-4-t0.0--gpt-4-t0.0',
    'gpt-4-0314-t0.0--gpt-4-0314-t0.0',
    'claude-2.1-t0.0--claude-2.1-t0.0',
    'gpt-4-t0.0--gpt-3.5-turbo-t0.0',
    'claude-2-t0.0--claude-2-t0.0',
    'gpt-3.5-turbo-1106-t0.0--gpt-3.5-turbo-1106-t0.0',
    'gpt-3.5-turbo-0613-t0.0--gpt-3.5-turbo-0613-t0.0',
]

top_10_models_new_clembench = [
'o1-preview-2024-09-12-t0.0--o1-preview-2024-09-12-t0.0',
 'gpt-4-0125-preview-t0.0--gpt-4-0125-preview-t0.0',
 'gpt-4-1106-preview-t0.0--gpt-4-1106-preview-t0.0',
 'gpt-4-turbo-2024-04-09-t0.0--gpt-4-turbo-2024-04-09-t0.0',
 'gpt-4-0613-t0.0--gpt-4-0613-t0.0',
 'claude-3-5-sonnet-20240620-t0.0--claude-3-5-sonnet-20240620-t0.0',
 'Meta-Llama-3.1-405B-Instruct-Turbo-t0.0--Meta-Llama-3.1-405B-Instruct-Turbo-t0.0',
 'gpt-4o-2024-05-13-t0.0--gpt-4o-2024-05-13-t0.0',
 'gpt-4o-2024-08-06-t0.0--gpt-4o-2024-08-06-t0.0',
 'claude-3-opus-20240229-t0.0--claude-3-opus-20240229-t0.0'
]

prompt_dpo_turn_wordle_no_explanation = """You are a language wizard who likes to guess words by using the given rules.\n\nWelcome to Wordle! You have six attempts to guess the target word, a valid English word of five lowercase letters (a-z). Please use the tags "guess:".\n\nFor instance, if your guess is "apple", your response should be\nguess: apple.\n\nAfter each guess, your answer will be validated, and you will receive feedback indicating which letters are correct (green), which letters are correct but in the wrong position (yellow), and which letters are incorrect (red). This feedback can be useful in determining which letters to include or exclude in your next guess.\n\nFor example, the feedback for "apple" might be:\nguess_feedback: a<yellow> p<yellow> p<green> l<yellow> e<red>\n\nLet\'s begin with your first guess."""
