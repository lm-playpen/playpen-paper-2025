
import re
import torch
import numpy as np
from playpen.agents.base_agent import Agent
from playpen.clemgame.clemgame import load_benchmark
from playpen.agents.clembench_agent import ClembenchAgent
from typing import Optional, Union, Dict, Any, Tuple, List
from playpen.backends.utils import ensure_alternating_roles
from transformers import PreTrainedModel, PreTrainedTokenizer


class GRPOAgent(Agent):  # interfaces model with playpen games
    reward: float

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            device: Optional[Union[str, int]] = None,
            output_split_prefix: Optional[str] = None  # not sure what this is... (copied from playpen code)
    ):
        super(GRPOAgent, self).__init__('grpo_agent')
        self.do_sample, self.temperature, self.max_new_tokens, self.min_length = False, None, 64, -1
        self.output_split_prefix, self.reward = output_split_prefix, None
        self.device = (0 if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model, self.tokenizer, self.top_k, self.top_p = model, tokenizer, None, None

    @property
    def generation_kwargs(self) -> Dict[str, Any]:
        return {
            'min_length': self.min_length,
            'pad_token_id': self.tokenizer.pad_token_id,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'max_new_tokens': self.max_new_tokens
        }

    def set_generation_kwargs(self, **generation_kwargs) -> None:
        assert set(generation_kwargs.keys()) <= set(self.generation_kwargs.keys())

        for k, v in generation_kwargs.items():
            assert hasattr(self, k), k
            setattr(self, k, v)

    def act(self) -> Tuple[Any, Any, str]:  # single game step (mostly copied from playpen code)
        current_messages = ensure_alternating_roles(self.observations)
        prompt_tokens = self.tokenizer.apply_chat_template(
            current_messages,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        prompt_tokens = prompt_tokens.to(self.device)
        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]
        prompt = {
            'inputs': prompt_text,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'return_full_text': False
        }

        model_output_ids = self.model.generate(prompt_tokens, **self.generation_kwargs)
        model_output = self.tokenizer.batch_decode(model_output_ids)[0]
        response_text = model_output.replace(prompt_text, '').strip()
        response = {'response': model_output}

        if self.output_split_prefix is not None:
            response_text = model_output.rsplit(self.output_split_prefix, maxsplit=1)[1]

        response_text = re.sub(self.tokenizer.eos_token, "", response_text)  # remove eos token string

        if response_text.endswith('||'):  # Llama model loves to output this for some reason...
            response_text = response_text[:-2].strip()

        return prompt, response, response_text

    def observe(self, observation, reward, *_) -> None:
        self.observations.append(observation)
        self.reward = reward

    def reset(self):
        self.observations.clear()
        self.reward = None

    def shutdown(self):  # abstract method of parent
        pass


class GameContainer:  # allows agent to play single instance of specified playpen game
    _reward: Optional[float]

    def __init__(
            self,
            name: str,
            instances: List[str],
            order: Union[str, Tuple[str, str]] = ('teacher', 'agent'),
            reward_from_agent: bool = True,
            max_reward: Optional[float] = None,  # max reward given by game scorer (for normalization)
            min_reward: Optional[float] = None  # min reward given by game scorer (for normalization)
    ):
        self.name, self.order, self._rwd_from_agent, self._player_idx = name, order, reward_from_agent, 1

        # orders model list for game
        if self.order == ('teacher', 'agent'):
            self._order = lambda a, t: [t, a]
            self._player_idx = 2
        elif self.order == ('agent', 'teacher'):
            self._order = lambda a, t: [a, t]
        elif self.order == 'agent':
            self._order = lambda a, _: [a]
        else:
            raise ValueError('order not in {(\'teacher\', \'agent\'), (\'agent\', \'teacher\'), \'agent\'}')
        if max_reward is None:
            assert min_reward is None
            self.max_reward, self._max_reward, self.min_reward = None, None, None
        else:
            assert min_reward < max_reward
            self.max_reward, self.min_reward, self._max_reward = max_reward, min_reward, max_reward - min_reward

        self.idx_map, self.benchmarks, self.experiment_configs, self.game_instances = [], [], [], []

        # aggregates/flattens all experiments in all instances*.json files into single list of idxs
        for inst in instances:
            self.benchmarks.append(load_benchmark(name, instances_name=inst))

            for exp in self.benchmarks[-1].instances['experiments']:
                self.experiment_configs.append(exp)
                self.game_instances.append(self.experiment_configs[-1].pop('game_instances'))
                self.idx_map.extend(  # maps idxs back to (instance_file, experiment, game_id) triples
                    (len(self.benchmarks) - 1, len(self.experiment_configs) - 1, i)
                    for i in range(len(self.game_instances[-1]))
                )

    def play(  # plays single game instance
            self,
            instance_id: int,
            agent: "GRPOAgent",
            teacher: ClembenchAgent,
            pos_reward: float = 1.0,
            neg_reward: float = 0.0,
            fail_reward: float = 0.0
    ) -> Tuple[List[Dict[str, str]], Optional[float]]:
        agent.reset()  # reset agent/teacher observations and reward
        teacher.reset()

        # map instance_id back to (instance_file, experiment, game_id) triple, then play game
        bench_idx, experiment_idx, gi_idx = self.idx_map[instance_id]
        game_master = self.benchmarks[bench_idx].create_game_master(
            self.experiment_configs[experiment_idx], self._order(agent, teacher)
        )
        game_master.setup(**self.game_instances[experiment_idx][gi_idx])
        game_master.play()

        gmi = game_master.interactions
        last_agent, agent_obs, reward = False, [], None

        for turn in gmi['turns']:
            for sub_turn in turn:
                if 'probe' not in sub_turn.get('action', {}).get('type', ''):  # privateshared :/
                    if sub_turn['from'] == f'Player {self._player_idx}' and sub_turn['to'] == 'GM':
                        agent_obs.append({'role': 'assistant', 'content': sub_turn['action']['content']})
                        last_agent = True
                    elif sub_turn['from'] == 'GM' and sub_turn['to'] == f'Player {self._player_idx}':
                        if isinstance(sub_turn['action']['content'], dict):
                            agent_obs.append({'role': 'user', 'content': sub_turn['action']['content']['content']})
                        else:
                            agent_obs.append({'role': 'user', 'content': sub_turn['action']['content']})
                    elif sub_turn['from'].startswith('Player ') and sub_turn['to'] == 'GM':
                        last_agent = False

        if len(agent_obs) > 0:
            agent_obs = ensure_alternating_roles(agent_obs)

        if any(
            'invalid' in x.get('action', {}).get('type', '') for x in gmi['turns'][-1] if x['to'] == x['from'] == 'GM'
        ) and not last_agent:  # aborted due to non-agent error
            return agent_obs, None

        if self._rwd_from_agent:
            reward = agent.reward
        elif len(teacher.rewards) > 0:
            reward = teacher.rewards[-1]
        if reward is None or np.isnan(reward):
            reward = float(fail_reward)
        elif self.max_reward is None or (pos_reward == self.max_reward and neg_reward == self.min_reward):
            reward = float(reward)  # if no max/min reward specified, can't normalize to [0, 1]
        else:  # normalize reward to the interval [neg_reward, pos_reward]
            reward = (((reward - self.min_reward) / self._max_reward) * (pos_reward - neg_reward)) + neg_reward

        return agent_obs, reward

    def __len__(self) -> int:
        return len(self.idx_map)
