import argparse
from cmath import inf
from datasets import Dataset
import pandas as pd
from collections import defaultdict
from huggingface_hub import login
from utils import games, top_10_models_old_clembench, top_10_models_new_clembench
import random

#TODO: reduce this with the following
def aborted_wordle(game, clembench_version):
    file_path_aborted_wordle_old = f'data/processed/turn_scores/{game}_old_witherrors.jsonl'
    file_path_aborted_wordle_new = f'data/processed/turn_scores/{game}_new_witherrors.jsonl'
    df_old = pd.read_json(file_path_aborted_wordle_old, lines=True)

    if clembench_version == 'old_and_new':
        #TODO: reduce in a function? with the data in the next
        df_new = pd.read_json(file_path_aborted_wordle_new, lines=True)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_old

    df_aborted = df[df.Aborted==1]
    df_aborted['chat'] = df_aborted['chat'].apply(
        lambda chat_list: list(map(lambda d: {k: v for k, v in d.items() if k != 'has_error'}, chat_list)))
    return df_aborted


def collect_old_and_new_files(game, clembench_version):
    file_path_old = f'data/processed/turn_scores/{game}_old_processed.jsonl'
    df_old = pd.read_json(file_path_old, lines=True)
    if clembench_version == 'old':
        df = df_old
    else:
        file_path_new = f'data/processed/turn_scores/{game}_new_processed.jsonl'
        df_new = pd.read_json(file_path_new, lines=True)
        df = pd.concat([df_old, df_new], ignore_index=True)
    return df

def instruction_splitter(row, first_message, splitting_word):
    last_index = first_message['content'].rfind(splitting_word)
    new_chat_list = [
        {'role': 'user', 'content': first_message['content'][:last_index]},
        {'role': 'user', 'content': first_message['content'][last_index:]}
    ]
    new_chat_list.extend(row['chat'][1:])
    return new_chat_list
  
def chat_processing(row, game):
    game_split_word_dict = {'taboo':'CLUE:', 'referencegame':'Expression:', 'wordle_withcritic':'guess:', 'imagegame':'Instruction:'}
    if game in ['wordle','wordle_withclue', 'privateshared']:
        return row['chat']
    word = game_split_word_dict[game]
    if row['chat']:
        first_message = row['chat'][0]
        if row['player'] == 'player 1':
            return row['chat']
        if game == 'taboo':
            if row['benchmark_version'] in ['v0.9', 'v1.0'] or first_message['content'].upper().count('CLUE') == 1:
                return row['chat']
        #assert game in ['referencegame', 'wordle_withcritic','imagegame']
        return instruction_splitter(row, first_message, word)

def remove_explanation(chat_list):
    if not isinstance(chat_list, list):
        return chat_list
    processed_chat = []
    for message in chat_list:
        if isinstance(message, dict) and 'content' in message:
            content = message['content']
            explanation_pos = content.lower().find('explanation:')
            if explanation_pos != -1:
                message = message.copy()  # Avoid modifying original
                message['content'] = content[:explanation_pos].strip()
        processed_chat.append(message)
    return processed_chat


def game_specific_chat_processing(game, df_successful, df_unsuccessful):
    df_successful['chat'] = df_successful.apply(lambda row: chat_processing(row, game), axis=1)
    df_unsuccessful['chat'] = df_unsuccessful.apply(lambda row: chat_processing(row, game), axis=1)
    return df_successful, df_unsuccessful

def first_turn_limit(other_turns_dict, turn_1_dict, game):
    sample_to_extract = 10000 if game not in ['wordle','wordle_withclue'] else 1000 #TODO: this solve the problem with wordle but still to much samples

    result_dict = defaultdict(list)
    for key in other_turns_dict:
        result_dict[key].extend(other_turns_dict[key])
    if turn_1_dict:
        indices = list(range(len(turn_1_dict['game'])))
        random.shuffle(indices)
        max_samples = min(sample_to_extract, len(indices))
        selected_indices = indices[:max_samples]
        for key in turn_1_dict:
            result_dict[key].extend([turn_1_dict[key][i] for i in selected_indices])
    return result_dict


def match_successful_unsuccessful_dialogue(game, df_successful_games, df_unsuccessful_games, match_count, model_condition):
    df_successful = df_successful_games.copy()
    df_unsuccessful = df_unsuccessful_games.copy()

    df_successful, df_unsuccessful = game_specific_chat_processing(game, df_successful, df_unsuccessful)

    df_successful['first_message'] = df_successful['chat'].apply(lambda x: x[0]['content'])
    df_unsuccessful['first_message'] = df_unsuccessful['chat'].apply(lambda x: x[0]['content'] if x else "")

    match_cols = ['game', 'game_id', 'benchmark_version', 'experiment', 'episode', 'first_message', 'target']
    match_cols += ['player'] if 'player' in df_successful_games.columns else []
    df_successful['merge_key'] = df_successful[match_cols].apply(lambda x: '_'.join(str(val) for val in x), axis=1)
    df_unsuccessful['merge_key'] = df_unsuccessful[match_cols].apply(lambda x: '_'.join(str(val) for val in x), axis=1)
    result_dict = defaultdict(list)
    for _, succ_row in df_successful.iterrows():
        matching_unsucc = df_unsuccessful[df_unsuccessful['merge_key'] == succ_row['merge_key']]
        if model_condition:
            if model_condition == 'best_models':
                #TODO: check this working
                if args.clembench_version == 'old':
                    matching_unsucc = matching_unsucc[matching_unsucc['model'].isin(top_10_models_old_clembench)]
                else:
                    #TODO: change checkin clembench verions, what is easies way?
                    matching_unsucc = matching_unsucc[matching_unsucc['model'].isin(list(set(top_10_models_old_clembench+top_10_models_new_clembench)))]
            elif model_condition == 'same_family_model':
                matching_unsucc = matching_unsucc[matching_unsucc['model'].str.contains(args.model_name, case=False, na=False)] #TODO: for mistral check also for 'mixtral'

        if match_count != float(inf): matching_unsucc = matching_unsucc.head(int(match_count))
        for _, unsucc_row in matching_unsucc.iterrows():
            result_dict['game'].append(succ_row['game'])
            result_dict['game_id'].append(succ_row['game_id'])
            result_dict['benchmark_version'].append(succ_row['benchmark_version'])
            result_dict['experiment'].append(succ_row['experiment'])
            result_dict['episode'].append(succ_row['episode'])
            result_dict['model_successful'].append(succ_row['model'])
            result_dict['model_unsuccessful'].append(unsucc_row['model'])
            result_dict['prompt'].append(succ_row['first_message'])
            result_dict['player'].append(succ_row['player'] if succ_row['player'] else 'player 1')
            result_dict['chosen'].append(succ_row['chat'])
            result_dict['rejected'].append(unsucc_row['chat'])
    return dict(result_dict)

def match_successful_unsuccessful_turn(game, df_successful_games, df_unsuccessful_games, match_count, model_condition):
    df_successful = df_successful_games.copy()
    df_unsuccessful = df_unsuccessful_games.copy()

    df_successful, df_unsuccessful = game_specific_chat_processing(game, df_successful, df_unsuccessful)

    match_cols = ['game', 'game_id', 'benchmark_version', 'experiment', 'episode', 'target']
    match_cols += ['player'] if 'player' in df_successful_games.columns else []
    df_successful['merge_key'] = df_successful[match_cols].apply(lambda x: '_'.join(str(val) for val in x), axis=1)
    df_unsuccessful['merge_key'] = df_unsuccessful[match_cols].apply(lambda x: '_'.join(str(val) for val in x), axis=1)
    result_dict = defaultdict(list)
    turn_1_dict = defaultdict(list)
    other_turns_dict = defaultdict(list)
    negatives_per_branches = defaultdict(int)
    for _, succ_row in df_successful.iterrows():
        matching_unsucc = df_unsuccessful[df_unsuccessful['merge_key'] == succ_row['merge_key']]
        if model_condition:
            if model_condition == 'best_models':
                #TODO: need to integrate different models depending on new or old
                if args.clembench_version == 'old':
                    matching_unsucc = matching_unsucc[matching_unsucc['model'].isin(top_10_models_old_clembench)]
                else:
                    #TODO: change checkin clembench verions, what is easies way?
                    matching_unsucc = matching_unsucc[matching_unsucc['model'].isin(list(set(top_10_models_old_clembench+top_10_models_new_clembench)))]
            elif model_condition == 'same_family_model':
                matching_unsucc = matching_unsucc[matching_unsucc['model'].str.contains(args.model_name, case=False, na=False)] #TODO: for mistral check also for 'mixtral'

        if game in ['wordle', 'wordle_withclue']:
            succ_row['chat'] = remove_explanation(succ_row['chat'])

        for idx in range(len(succ_row['chat'])):
            if succ_row['chat'][idx]['role'] == 'assistant':
                for _, unsucc_row in matching_unsucc.iterrows():
                    if game in ['wordle', 'wordle_withclue']:
                        unsucc_row['chat'] = remove_explanation(unsucc_row['chat'])
                    if unsucc_row['chat'] and len(unsucc_row['chat']) >= len(succ_row['chat'][:idx]) and unsucc_row['chat'][:len(succ_row['chat'][:idx])] == \
                            succ_row['chat'][:idx]:  # all the chat history should be shared
                        if succ_row['chat'][idx] != unsucc_row['chat'][idx]:  # the 'assistant' turn should instead be different
                            if idx > 0:
                                negatives_per_branches[idx] += 1

                            # Create a sample dictionary
                            sample = {
                                'game': succ_row['game'],
                                'game_id': succ_row['game_id'],
                                'benchmark_version': succ_row['benchmark_version'],
                                'experiment': succ_row['experiment'],
                                'episode': succ_row['episode'],
                                'model_successful': succ_row['model'],
                                'model_unsuccessful': unsucc_row['model'],
                                'prompt': succ_row['chat'][:idx],
                                'player': succ_row['player'] if 'player' in succ_row and succ_row[
                                    'player'] else 'player 1',
                                'branch_turn': idx,
                                'chosen': [succ_row['chat'][idx]],
                                'rejected': [unsucc_row['chat'][idx]]
                            }

                            if args.first_turn_limit:
                                if idx == 1:
                                    for key, value in sample.items():
                                        turn_1_dict[key].append(value)
                                else:
                                    for key, value in sample.items():
                                        other_turns_dict[key].append(value)
                            else:
                                for key, value in sample.items():
                                    result_dict[key].append(value)


    print(f'Game: {game}: ', negatives_per_branches)

    if args.first_turn_limit:
        result_dict = first_turn_limit(other_turns_dict, turn_1_dict, game)

    return dict(result_dict)

def matches_all_files(match_count, model_condition):
    success_and_lose_all_games = defaultdict(list)
    preference_depth_function = {'dialogue':match_successful_unsuccessful_dialogue, 'turn':match_successful_unsuccessful_turn}

    for game in games:
        #TODO: delete this coondition when referencegame in turn_scores file as well (and not repeat the line df = pd.read_json(file_path, lines=True))
        if game != 'referencegame':
            df = collect_old_and_new_files(game, clembench_version=args.clembench_version)
        else:
            file_path = f'data/processed/turn_scores/referencegame/{game}_new_processed_10.jsonl'
            df = pd.read_json(file_path, lines=True)    #TODO: reduce this line

        df_success = df[df.Success == 1]
        df_lose = df[df.Lose == 1]
        game_matches = preference_depth_function[args.preference_depth](game, df_success, df_lose, match_count, model_condition)
        if args.aborted_interactions:
            if 'wordle' in game:    #wordle and variants have their own files where aborted are stored due to a processing differences
                df_aborted = aborted_wordle(game, clembench_version=args.clembench_version)
            else:
                df_aborted = df[df.Aborted == 1]
            game_matches_aborted = preference_depth_function[args.preference_depth](game, df_success, df_aborted, match_count, model_condition)
            #TODO: put this if condition in the line above or simplify
            if game_matches_aborted:
                game_matches = {key: game_matches[key] + game_matches_aborted[key] for key in game_matches} #if because for referencegame we do not have data from the same-model family (all from 10 best models)
        for key in game_matches:
            success_and_lose_all_games[key].extend(game_matches[key])
    return dict(success_and_lose_all_games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create DPO datasets')
    parser.add_argument('--clembench_version', default='old', choices=['old', 'old_and_new'], help='clembench versions employed')
    parser.add_argument('--preference_depth', default='dialogue', choices=['dialogue', 'turn'], help='level of depth for the preference dataset, dialogue has positive and good samples that are all conversation, turn is only the following turn')
    parser.add_argument('--neg', default=float(inf), type=float, help='number of negative samples per every positive one')
    parser.add_argument('--model_condition', default=False, choices=[False, 'best_models', 'same_family_model'], help='restriction to the negative samples to be from best models or from the family of the model to train (llama)')
    parser.add_argument('--model_name', default='llama', choices=['llama', 'mistral'], help='base model name for the same family model condition')
    parser.add_argument('--aborted_interactions', default=True, choices=[True, False], help='integrating aborted interactions as negative samples')
    parser.add_argument('--first_turn_limit', default=True, choices=[True, False], help='if there is a turn limit, only first 10K samples considered for the 1st turn per game')
    #TODO: take this out in common with DPO_training.py and KTO_training.py
    parser.add_argument('--hf_login', default="", help='hf login token')
    parser.add_argument('--hf_repo', default='', help='huggingface repository to store the created datasets')
    args = parser.parse_args()

    login(f"{args.hf_login}")
    success_and_lose_all_games = matches_all_files(match_count=args.neg, model_condition=args.model_condition)
    hf_dataset = Dataset.from_dict(success_and_lose_all_games)
    hf_dataset.push_to_hub(f"{args.hf_repo}/DPO_{args.preference_depth}_{int(args.neg) if args.neg != float(inf) else 'all'}neg{'_'+args.model_condition if args.model_condition else ''}_{args.clembench_version}_klimit")
