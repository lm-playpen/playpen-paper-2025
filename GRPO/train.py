
import os
import json
import torch
import pickle
import random
from traceback import format_tb
from peft import LoraConfig, set_peft_model_state_dict
from training import (
    LORA_LAYER_MAP,
    GRPOTrainer,
    GRPOConfig,
    ParallelModel,
    eval_model,
    grpo_arg_set,
    chkpt_name,
    peft_sd,
    sd_to_cpu
)


def eval_and_checkpoint(ep_n, step_n, header=''):
    eval_model(grpo_trainer, save_file, test_rewards, test_accs, header=header)

    if save_model:  # checkpoint if flag is set
        chkpt_fp = chkpt_name(save_fp)
        chkpt_fp0, chkpt_fp1 = chkpt_fp.rsplit('/', 1)
        chkpt_fp1 = chkpt_fp1.strip('/')

        if chkpt_fp1 not in os.listdir(chkpt_fp0):  # create dir for checkpoints for this run if it doesn't exist
            os.mkdir(chkpt_fp)

        chkpt_fp = f'{chkpt_fp}/{ep_n}_{step_n}/'.replace('//', '/')  # dir for this specific checkpoint
        os.mkdir(chkpt_fp)

        if args.get('lora', False):
            model.modules[0].model.model.save_pretrained(f'{chkpt_fp}model')  # checkpointing LoRA weights
        else:
            with open(f'{chkpt_fp}model', 'wb') as f_chkpt:  # checkpointing non-LoRA model (entire state dict)
                pickle.dump(model.modules[0].model.model.state_dict(), f_chkpt)

        with open(f'{chkpt_fp}optimizer', 'wb') as f_chkpt:  # checkpointing optimizer
            pickle.dump(grpo_trainer.optimizer.state_dict(), f_chkpt)

        with open(f'{chkpt_fp}idxs', 'w') as f_chkpt:  # save training batch order (for checkpointing mid-epoch)
            json.dump(train_idxs, f_chkpt)

        with open(save_fp, 'w') as f_save:  # save stats file (contains args for re-loading model/other hyperparams)
            json.dump(save_file, f_save)


# parsing command-line args
args, save_file, save_fp, load_chkpt, save_model, dpm = grpo_arg_set.parse_args(os.path.realpath(__file__))

if args.get('seed', None) is not None:  # set seed if applicable
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

grpo_config = GRPOConfig(
    pos_reward=args['pos_reward'],
    neg_reward=args['neg_reward'],
    fail_reward=args['fail_reward'],
    grad_acc_batch_size=args['grad_acc_batch_size'],
    batch_size=args['batch_size'],
    group_size=args['group_size'],
    beta=(None if args['beta'] == -1 else args['beta']),
    mask_system=(not args.get('no_mask', False)),
    retry_limit=args['retry_limit'],
    generation_kwargs={
        'min_length': -1,
        'do_sample': True,
        'top_k': args['top_k'],
        'top_p': args['top_p'],
        'temperature': args['temperature'],
        'max_new_tokens': args['max_new_tokens']
    },
    no_privateshared=args.get('no_ps', False),
    both_roles=args.get('both_roles', False)
)

if args.get('lora', False):
    lora_config = LoraConfig(
        r=args['lora_r'],
        lora_alpha=args['lora_alpha'],
        lora_dropout=args['lora_dropout'],
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=LORA_LAYER_MAP[args['lora_layers']]
    )
    ref_model = None  # can just turn off adapters to get reference logits
else:
    lora_config = None
    ref_model = ParallelModel.from_pretrained(  # need second model to get reference logits
        args['model_id'],
        grpo_config,
        {'token': args['access_token']},
        args['teacher_model'],
        {'temperature': args['teacher_temperature'], 'max_tokens': args['teacher_max_tokens']}
    )

model = ParallelModel.from_pretrained(  # policy model (i.e. the one being trained)
    args['model_id'],
    grpo_config,
    {'token': args['access_token']},
    args['teacher_model'],
    {'temperature': args['teacher_temperature'], 'max_tokens': args['teacher_max_tokens']},  # teacher model gen. args.
    lora_config=lora_config,
    devices_per_model=dpm
)
optimizer = torch.optim.Adam(model.parameters(), lr=args['learn_rate'])

if load_chkpt is None:  # starting from epoch 0, step 0
    test_rewards, test_accs, running_accs, running_scores, running_scores_parallel = [], [], [], [], []
    init_ep, init_step, train_idxs = 0, -1, None
else:  # loading from checkpoint
    # finding most recent checkpoint for specified run
    all_chk = [tuple(map(int, x.split('_'))) for x in os.listdir(load_chkpt)]
    init_ep = max(x for x, _ in all_chk)
    init_step = max(y for x, y in all_chk if x == init_ep)
    load_fp, ep_key_dict, ep_keys = f'{load_chkpt}/{init_ep}_{init_step}/', {}, []

    if args['lora']:  # loading checkpointed LoRA weights
        set_peft_model_state_dict(model.modules[0].model.model, peft_sd(f'{load_fp}model'))
    else:
        with open(f'{load_fp}model', 'rb') as f:  # loading checkpointed model weights
            model.modules[0].model.model.load_state_dict(pickle.load(f))

    model.set_mode('train')  # update model weights across processes (kinda hacky)
    model._sds_aligned = False
    model.set_mode('gen')

    with open(f'{load_fp}optimizer', 'rb') as f:  # loading optimizer weights
        optimizer.load_state_dict(pickle.load(f))

    with open(f'{load_fp}idxs', 'r') as f:
        train_idxs = json.load(f)  # to maintain order

    for k in list(save_file['ep_data'].keys()):  # removing stats (e.g. loss) past checkpoint
        k_ep, k_step = map(int, k.split(';'))

        if k_ep > init_ep or (k_ep == init_ep and k_step > init_step):
            save_file['ep_data'].pop(k)
        elif k_ep in ep_key_dict.keys():
            ep_key_dict[k_ep].append(k_step)
        else:
            ep_key_dict.update({k_ep: [k_step]})

    for k in sorted(list(ep_key_dict.keys())):  # sorting epoch;step keys display
        ep_key_dict[k].sort()
        ep_keys.extend(f'{k};{v}' for v in ep_key_dict[k])

    # update running accuracy list, etc. (for display)
    running_accs = [round(save_file['ep_data'][k]['stats']['accuracy'], 3) for k in ep_keys]
    running_scores = [round(save_file['ep_data'][k]['stats']['mean_rewards'], 3) for k in ep_keys]
    running_scores_parallel = [round(save_file['ep_data'][k]['stats']['parallel_scores'], 3) for k in ep_keys]
    test_accs = [round(x['accuracy'], 3) for x in save_file['test']['steps']]
    test_rewards = [round(x['mean_reward'], 3) for x in save_file['test']['steps']]

grpo_trainer = GRPOTrainer(
    model,
    torch.optim.Adam(model.parameters(), lr=args['learn_rate']),
    grpo_config,
    ref_model=ref_model
)

if train_idxs is None:  # i.e. starting from scratch (not loading checkpoint)
    train_idxs = grpo_trainer.instance_idxs()
    # (taboo, 53) causes error
    train_idxs = [x for x in train_idxs if not (x == ('taboo', 53) or x == ('taboo_teacher', 53))]
    random.shuffle(train_idxs)

n_steps = len(train_idxs) // args['batch_size']  # steps per epoch

if args['eval_interval'] == 0:  # eval at the end of each epoch
    save_file['test']['interval'] = n_steps

if load_chkpt is None and args.get('initial_eval', False):  # get baseline eval. rewards if applicable
    eval_model(grpo_trainer, save_file, test_rewards, test_accs, header='INITIAL')

try:
    for i in range(init_ep, args['epochs']):  # main training loop
        for step in range(init_step + 1, n_steps):
            print(f'EPOCH {i}, STEP {step}:\n\n')

            # training step
            step_stats = grpo_trainer.step(train_idxs[step * args['batch_size']:(step + 1) * args['batch_size']])
            step_str = f'{i};{step}'

            # update stats file
            save_file['ep_data'].update({step_str: {'query_resp_reward': step_stats.pop('query_resp_reward')}})
            save_file['ep_data'][step_str].update({'stats': step_stats})
            running_accs.append(round(step_stats['accuracy'], 3))
            running_scores.append(round(step_stats['mean_rewards'], 3))
            running_scores_parallel.append(round(step_stats['parallel_scores'], 3))

            # display step trajectories + stats
            print()
            print(
                '\n----------------------------------------------------------\n'.join(
                    f'{q}{t} ({r})' for q, t, r in save_file['ep_data'][step_str]['query_resp_reward']
                )
            )
            print('\n----------------------------------------------------------')
            print('\n\n')

            print(f'GRPO Loss = {step_stats["loss"]}')
            print(f'Reward = {step_stats["mean_rewards"]}')
            print(f'Response Len. = {step_stats["response_len"]}')
            print(f'KL = {step_stats["kl"]}')
            print(f'Test Rewards: {test_rewards}')
            print(f'Test Rewards (ACC): {test_accs}')
            print(f'Running Rewards: {running_scores}')
            print(f'Running Acc: {running_accs}')
            print(f'Running Acc (PARALLEL): {running_scores_parallel}\n\n\n')

            if args['eval_interval'] > 0 and (step + 1) % args['eval_interval'] == 0:  # evaluation step
                eval_and_checkpoint(i, step)

            with open(save_fp, 'w') as f:  # save stats file
                json.dump(save_file, f)

        random.shuffle(train_idxs)  # randomize training idxs at the end of each epoch
        init_step = -1

        if args['eval_interval'] == 0:  # eval every epoch
            eval_and_checkpoint(i + 1, -1)
except Exception as e:  # add traceback to stats file
    save_file.update({'error': f'{type(e)}: \n\n' + '\n'.join(format_tb(e.__traceback__))})

    with open(save_fp, 'w') as f:
        json.dump(save_file, f)

    raise e

if not args['eval_interval'] == 0:  # do a final eval
    eval_and_checkpoint(args['epochs'], -1, header='FINAL')
