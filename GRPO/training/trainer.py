
import gc
import torch
import random
from itertools import chain
from training.utils import tqdm_iter
from .parallelization import ParallelModel
from training.playpen_interface import GameContainer
from transformers import DataCollatorForLanguageModeling
from typing import Optional, Dict, Union, Tuple, List, Any


class GRPOConfig:  # config for GRPOTrainer/ParallelModel
    def __init__(
            self,
            pos_reward: float = 1.0,  # max. reward
            neg_reward: float = 0.0,  # min. reward
            fail_reward: float = 0.0,  # penalty (reward) for aborted episode (e.g. incorrect response format)
            grad_acc_batch_size: int = 1,  # gradient accumulation batch size (for update step)
            batch_size: int = 16,  # step size (# of trajectories per weight update)
            group_size: int = 8,  # number of completions per instance
            mask_system: bool = True,  # mask game master/teacher responses when computing loss
            retry_limit: int = 1,  # number of retry attempts when teacher messes up
            beta: Optional[float] = 0.04,  # beta value in GRPO loss
            generation_kwargs: Optional[Dict[str, Any]] = None,
            no_privateshared: bool = False,  # don't include privateshared instances
            both_roles: bool = False
    ):
        self.actual_batch_size = batch_size * group_size
        self.pos_reward, self.neg_reward, self.fail_reward = pos_reward, neg_reward, fail_reward
        self.beta, self.mask_system, self.retry_limit = beta, mask_system, retry_limit
        self.batch_size, self.group_size, self.grad_acc_batch_size = batch_size, group_size, grad_acc_batch_size
        self.n_batch = self.actual_batch_size // self.grad_acc_batch_size
        self.generation_kwargs = {} if generation_kwargs is None else generation_kwargs
        self.no_privateshared, self.both_roles = no_privateshared, both_roles

        assert self.actual_batch_size % self.grad_acc_batch_size == 0
        assert self.beta is None or self.beta >= 0.0
        assert self.pos_reward > self.neg_reward
        assert self.retry_limit >= 1
        assert self.group_size > 1


class GRPOTrainer:
    benchmarks: Dict[bool, Dict[str, "GameContainer"]]

    def __init__(
            self,
            model: ParallelModel,
            optimizer: torch.optim.Optimizer,
            config: "GRPOConfig",
            lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            ref_model: Optional[ParallelModel] = None
    ):
        self.model, self.config = model, config
        self.ref_model, self.optimizer, self.lr_scheduler = ref_model, optimizer, lr_scheduler
        self.data_collator = DataCollatorForLanguageModeling(self.model.modules[0].model.tokenizer, mlm=False)
        self.optimizer.zero_grad()

    def instance_idxs(self, train: bool = True) -> List[Tuple[str, int]]:
        # returns (game_name, index) pairs for train/test set
        return self.model.modules[0].instance_idxs(train=train)

    def eval(self, progress_bar: bool = True) -> Dict[str, Union[float, Dict[str, Union[float, Dict[int, str]]]]]:
        # validation step
        gc.collect()
        torch.cuda.empty_cache()
        eval_stats = {  # per-game stats
            k: {'responses': {}, 'mean_reward': 0.0, 'accuracy': 0.0}
            for k in self.model.modules[0].benchmarks[False].keys()
        }
        gen_outputs = self.model.eval(progress_bar=progress_bar)  # gets eval trajectories

        for (game, idx), (response, mask, reward, query, dialog) in gen_outputs.items():  # update stats
            if response is None:  # teacher screwed up
                rwd = ''
            else:
                eval_stats[game]['accuracy'] += abs(reward.item() - self.config.pos_reward) < 0.01
                eval_stats[game]['mean_reward'] += reward.item()
                rwd = reward.item()

            eval_stats[game]['responses'].update({idx: f'{query}{dialog} ({rwd})'})

        gc.collect()
        torch.cuda.empty_cache()
        total_examples, total_reward, total_acc = 0, 0.0, 0.0

        for v in eval_stats.values():
            n_valid = sum(1 for x in v['responses'].values() if not x.endswith('()'))  # non-aborted (bc of teacher)
            total_examples += n_valid
            total_reward += v['mean_reward']
            total_acc += v['accuracy']

            v['mean_reward'] /= n_valid
            v['accuracy'] /= n_valid

        return {
            'games': eval_stats,
            'accuracy': total_acc / total_examples,
            'mean_reward': total_reward / total_examples
        }

    def step(  # generate trajectories & update model
            self,
            batch: List[Tuple[str, int]],  # (game_name, index) pairs
            progress_bar: bool = True  # display tqdm progress bar
    ) -> Dict[str, Union[float, List[Tuple[str, str, float]]]]:
        assert len(batch) == self.config.batch_size
        torch.cuda.empty_cache()
        gc.collect()
        stats = {  # stats to return
            'query_resp_reward': [],
            'parallel_scores': 0.0,
            'mean_rewards': 0.0,
            'response_len': 0.0,
            'accuracy': 0.0,
            'loss': 0.0,
            'kl': 0.0
        }
        advantages, input_ids, masks, aborted_groups, max_mask, n_ep_aborted = [], [], [], set(), 0, 0

        with torch.no_grad():  # compute trajectories & reference logits
            for group in self.model.generate(batch, progress_bar=progress_bar).values():  # computing trajectories
                rewards, g_masks, g_ids, g_ps = [], [], [], False

                for resp, mask, rwd, query, dialog in group:  # group = {completions for single instance)
                    stats['query_resp_reward'].append((query, dialog, (rwd if rwd is None else rwd.item())))
                    g_masks.append(mask)
                    rewards.append(rwd)

                    if resp is None:  # aborted due to teacher error
                        n_ep_aborted += 1
                        g_ids.append(None)
                    else:
                        g_ids.append({'input_ids': resp, 'attention_mask': torch.ones_like(resp)})
                        stats['mean_rewards'] += rwd.item()
                        corr = abs(rwd.item() - self.config.pos_reward) < 0.01
                        stats['accuracy'] += corr
                        g_ps = g_ps or corr  # "group parallel score" = at least 1 completion got max reward

                        max_mask = max(max_mask, mask.size(0))
                        stats['response_len'] += mask.size(0)

                valid_idxs = [i for i, r in enumerate(rewards) if r is not None]  # non-aborted (teacher) episodes

                if len(valid_idxs) == 0:  # aborted group (all episodes aborted)
                    aborted_groups.add(len(advantages))
                    advantages.append(None)
                else:
                    if len(valid_idxs) < self.config.group_size:
                        # replace teacher-aborted episodes w/ random valid episodes from same group (best I could do)
                        for i in range(len(rewards)):
                            if rewards[i] is None:  # aborted episode
                                replace_idx = random.choice(valid_idxs)
                                rewards[i] = rewards[replace_idx].clone()
                                g_ids[i] = {
                                    'input_ids': g_ids[replace_idx]['input_ids'].clone(),
                                    'attention_mask': g_ids[replace_idx]['attention_mask'].clone()
                                }
                                g_masks[i] = g_masks[replace_idx].clone()

                    group_reward = torch.stack(rewards)
                    mean_reward = group_reward.mean(dim=0)
                    std_reward = group_reward.std(dim=0)
                    advantages.append(  # computing GRPO advantages
                        torch.zeros_like(group_reward) if std_reward.item() == 0.0 else
                        ((group_reward - mean_reward) / std_reward)
                    )

                stats['parallel_scores'] += g_ps
                input_ids.append(g_ids)
                masks.append(g_masks)

            if len(aborted_groups) > 0:
                if len(aborted_groups) == self.config.batch_size:  # every single episode was aborted by teacher
                    raise Exception('This is bad...')

                valid_groups = [i for i in range(len(advantages)) if i not in aborted_groups]

                for i in aborted_groups:
                    # if every episode in group was aborted, replace w/ random valid group (again, best I could do)
                    replace_idx = random.choice(valid_groups)
                    advantages[i] = advantages[replace_idx].clone()
                    input_ids[i] = [
                        {'input_ids': x['input_ids'].clone(), 'attention_mask': x['attention_mask'].clone()}
                        for x in input_ids[replace_idx]
                    ]
                    masks[i] = [m.clone() for m in masks[replace_idx]]

            masks = torch.stack(tuple(  # concatenating all masks
                torch.cat((
                    torch.zeros((max_mask - m.size(0),), device=self.model.modules[0].device, dtype=m.dtype),
                    m.to(device=self.model.modules[0].device)
                ))
                for m in chain.from_iterable(masks)
            ))
            advantages = torch.cat(advantages).to(device=self.model.modules[0].device)  # concatenating all advantages
            input_data = self.data_collator(list(chain.from_iterable(input_ids)))  # collating all inputs
            input_data.pop('labels', None)

            # add one padding token to each input sequence (simplifies logit computation)
            pad_ids = torch.full(
                (input_data['input_ids'].size(0), 1),
                self.model.modules[0].model.tokenizer.pad_token_id,
                dtype=input_data['input_ids'].dtype,
                device=input_data['input_ids'].device,
            )
            pad_masks = torch.zeros(
                (input_data['attention_mask'].size(0), 1),
                dtype=input_data['attention_mask'].dtype,
                device=input_data['attention_mask'].device,
            )
            input_data['input_ids'] = torch.cat((pad_ids, input_data['input_ids']), dim=1)
            input_data['attention_mask'] = torch.cat((pad_masks, input_data['attention_mask']), dim=1)

            if self.config.beta is None:  # don't compute reference logits if ignoring KL penality (i.e. beta = None)
                ref_logit_list, stats['kl'] = [None] * self.config.n_batch, -1.0
            elif self.ref_model is None:  # LoRA model: just compute reference logits w/ adapters switched off
                ref_logit_list = self.model.ref_logits(input_data, max_mask, progress_bar)
            else:  # non-lora model: use separate model for reference logits
                ref_logit_list = self.ref_model.ref_logits(input_data, max_mask, progress_bar)

        forward_pass = tqdm_iter(  # generator: yields policy model logits
            self.config.n_batch,
            generator=enumerate(zip(self.model.train(input_data, max_mask), ref_logit_list)),
            progress_bar=progress_bar,
            desc='Optimizing'
        )

        for i, (policy_logits, ref_logits) in forward_pass:  # compute GRPO loss w/ gradient accumulation
            start, end = i * self.config.grad_acc_batch_size, (i + 1) * self.config.grad_acc_batch_size
            ref_logits = ref_logits.to(device=policy_logits.device)
            iter_advs = advantages[start:end].unsqueeze(1).to(device=policy_logits.device)
            iter_mask = masks[start:end, :].to(device=policy_logits.device)
            token_loss = torch.exp(policy_logits - policy_logits.detach()) * iter_advs  # GRPO loss

            if ref_logits is not None:  # KL penalty
                token_kl = torch.exp(ref_logits - policy_logits) - (ref_logits - policy_logits) - 1
                stats['kl'] += ((token_kl * iter_mask).sum(dim=1) / iter_mask.sum(dim=1)).sum().item()

                if self.config.beta > 0.0:
                    # beta = None: don't compute ref. logits; beta = 0.0: compute, but don't use (i.e. for stats only)
                    token_loss = token_loss - (self.config.beta * token_kl)

            # GRPO loss w/ game master/teacher mask (if applicable)
            loss = ((-token_loss * iter_mask).sum(dim=1) / iter_mask.sum(dim=1)).mean() / self.config.n_batch
            stats['loss'] += loss.item()
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # mean stats, ignoring teacher-aborted episodes
        stats['mean_rewards'] /= self.config.actual_batch_size - n_ep_aborted
        stats['response_len'] /= self.config.actual_batch_size - n_ep_aborted
        stats['accuracy'] /= self.config.actual_batch_size - n_ep_aborted
        stats['parallel_scores'] /= self.config.batch_size - len(aborted_groups)
        stats['kl'] /= self.config.actual_batch_size - n_ep_aborted

        return stats
