
import gc
import json
import torch
import random
import warnings
from itertools import chain
from playpen.clemgame import benchmark  # just to load model registry (for teacher model)
from transformers import PreTrainedModel
from playpen.backends import ModelSpec, get_model_for
from playpen.agents.clembench_agent import ClembenchAgent
from transformers import AutoModelForCausalLM, AutoTokenizer
from training.playpen_interface import GRPOAgent, GameContainer
from .utils import set_batch_size, tqdm_iter, null_context, peft_sd, sd_to_cpu
from peft import LoraConfig, PeftModel, set_peft_model_state_dict, get_peft_model
from typing import TYPE_CHECKING, Optional, Dict, Any, Union, Generator, Tuple, List, Iterable


if TYPE_CHECKING:
    from .trainer import GRPOConfig
else:
    GRPOConfig = None

_generate_output_type = Tuple[
    Optional[torch.LongTensor],
    Optional[torch.LongTensor],
    Optional[torch.FloatTensor],
    str,
    str
]
_clm_mistral = 'clembench-playpen/Mistral-Small-24B-Instruct-2501_playpen_SFT_DFINAL_0.6K-steps'
_clm_mistral_merged = 'clembench-playpen/Mistral-Small-24B-Instruct-2501_playpen_SFT_merged_fp16_DFINAL_0.6K-steps'
_clm_llama = 'clembench-playpen/llama-3.1-8B-Instruct'
_us_llama = 'unsloth/meta-llama-3.1-8b-instruct-bnb-4bit'


class ParallelModel:  # wraps multiple copies of the model for parallel generation/ref. logit computation
    def __init__(self, modules: List["ParallelModule"], config: "GRPOConfig", _fpt: bool = False):
        self.modules, self.config, self.mode, self._sds_aligned = modules, config, '', _fpt
        self.is_peft_model = self.modules[0].is_peft_model
        self.modules[0]._is_main = True  # "main" module does training
        self.set_mode('gen')

        for m in self.modules[1:]:
            m._is_main = False

    def set_mode(self, mode: str, **kwargs) -> None:  # sets mode to "fwd" (ref. logits), "eval", "gen", or "train"
        if not mode == self.mode:
            self.mode = mode

            if mode == 'train':  # only main module trains
                self.modules[0].set_mode('train')
            else:
                if len(self.modules) > 1 and not self._sds_aligned:
                    # copying main state dict after weight update
                    if self.modules[0].is_peft_model:
                        sd = peft_sd(self.modules[0])
                    else:
                        sd = sd_to_cpu(self.modules[0].model.model.state_dict())
                else:
                    sd = None

                for i, m in enumerate(self.modules):
                    if i > 0 and not self._sds_aligned:
                        # sharing main state dict with other copies after weight update
                        if m.is_peft_model:
                            set_peft_model_state_dict(m.model.model, sd)
                        else:
                            m.load_state_dict(sd)

                    m.set_mode(mode, **kwargs)

    def parameters(self) -> Iterable[torch.nn.parameter.Parameter]:
        return self.modules[0].model.model.parameters()

    def train(
            self,
            inputs: Dict[str, torch.Tensor],
            n_logits_to_keep: int
    ) -> Generator[torch.Tensor, None, None]:
        self.set_mode('train')
        self._sds_aligned = False  # so it knows to share main state dict w/ other workers

        yield from self.modules[0].forward(inputs, n_logits_to_keep)

    def ref_logits(  # computes ref. logits in parallel
            self,
            inputs: Dict[str, torch.Tensor],
            n_logits_to_keep: int,
            progress_bar: bool
    ) -> List[torch.Tensor]:
        self.set_mode('fwd')
        input_size = -(-inputs['input_ids'].size(0) // len(self.modules))
        model_inputs = [  # splitting inputs across workers
            ({k: v[i * input_size:(i + 1) * input_size] for k, v in inputs.items()}, n_logits_to_keep, progress_bar)
            for i in range(len(self.modules))
        ]
        model_outputs = torch.nn.parallel.parallel_apply(self.modules, model_inputs)  # logits to output
        model_outputs = torch.cat(tuple(torch.cat(m, dim=0).to(self.modules[0].device) for m in model_outputs), dim=0)

        # split to zip with policy logits in training loop
        return list(torch.split(model_outputs, self.config.grad_acc_batch_size, dim=0))

    def eval(self, progress_bar: bool = True) -> Dict[Tuple[str, int], _generate_output_type]:
        # parallel validation step
        self.set_mode('eval')
        eval_idxs = self.modules[0].instance_idxs(train=False)
        group_outputs = self._parallel_apply_gen(eval_idxs, progress_bar, train=False)

        return {k: v[0] for k, v in group_outputs.items()}

    def generate(  # parallel trajectory generation step
            self,
            game_idxs: List[Tuple[str, int]],
            progress_bar: bool
    ) -> Dict[Tuple[str, int], List[_generate_output_type]]:
        self.set_mode('gen')

        return self._parallel_apply_gen(game_idxs, progress_bar, rf=self.config.group_size)

    def _parallel_apply_gen(  # shared by ParallelModel.eval() and ParallelModel.generate()
            self,
            game_idxs: List[Tuple[str, int]],
            progress_bar: bool,
            rf: int = 1,  # replication factor
            train: bool = True
    ) -> Dict[Tuple[str, int], List[_generate_output_type]]:
        # shuffling helps ensure max. utilization of all workers (e.g. so one worker doesn't get all the long games)
        game_idxs = game_idxs * rf
        random.shuffle(game_idxs)

        n_per_module = -(-len(game_idxs) // len(self.modules))
        group_inputs = [  # splitting inputs across workers
            (game_idxs[i * n_per_module:(i + 1) * n_per_module], progress_bar, train)
            for i in range(len(self.modules))
        ]
        group_outputs = torch.nn.parallel.parallel_apply(self.modules, group_inputs)  # generated trajectories
        out_data = group_outputs[0]

        for x in group_outputs[1:]:  # organize trajectories by group (i.e. instance)
            for k, v in x.items():
                if k in out_data.keys():
                    out_data[k].extend(v)
                else:
                    out_data.update({k: v})

        return out_data

    @classmethod
    def from_pretrained(  # instantiates from pretrained model
            cls,
            model_id: str,
            config: "GRPOConfig",
            model_kwargs: Optional[Dict[str, Any]] = None,
            teacher_model: str = 'gpt-4o-mini-2024-07-18',
            teacher_kwargs: Optional[Dict[str, Any]] = None,
            lora_config: Optional[LoraConfig] = None,
            devices_per_model: int = 1  # num. of GPUs per model copy
    ) -> "ParallelModel":
        teacher_kwargs = {} if teacher_kwargs is None else teacher_kwargs
        model_kwargs = {} if model_kwargs is None else model_kwargs
        assert 'device_map' not in model_kwargs
        token = model_kwargs.pop('token', None)
        modules = []

        if model_id.startswith('clembench-playpen/'):  # hacky stuff for loading playpen SFT models
            def load_model(dvmap):
                if model_id == _clm_mistral:
                    mdl = AutoModelForCausalLM.from_pretrained(_clm_mistral, device_map=dvmap, **model_kwargs)
                    mdl.bfloat16().cuda()
                elif model_id.startswith(_clm_llama):
                    mdl = AutoModelForCausalLM.from_pretrained(_us_llama, device_map=dvmap, **model_kwargs)
                    mdl = PeftModel.from_pretrained(mdl, model_id)
                    mdl = mdl.merge_and_unload()
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise NotImplementedError

                return mdl
        else:
            def load_model(dvmap):
                return AutoModelForCausalLM.from_pretrained(model_id, token=token, device_map=dvmap, **model_kwargs)

        def add_module(mdl, dev):  # adds model clone to module list
            if lora_config is not None:
                mdl = get_peft_model(mdl, lora_config)

            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
            tokenizer.pad_token = tokenizer.bos_token
            agent = GRPOAgent(mdl, tokenizer, device=dev)
            modules.append(ParallelModule(  # instantiate worker model
                agent,
                dev,
                teacher_model,
                teacher_kwargs,
                config,
                (lora_config is not None)
            ))

            if model_id.startswith('clembench-playpen/'):  # more hacky stuff for loading playpen SFT models
                modules[-1]._header_tok = modules[-1].model.tokenizer.bos_token_id

                if model_id == _clm_mistral:
                    modules[-1]._user_header, modules[-1]._assistant_header = (2, 3), (4,)
                    modules[-1]._user_header_len, modules[-1]._assistant_header_len = 2, 1
                    modules[-1]._min_header_len = 1
                elif model_id == _clm_llama:
                    modules[-1]._user_header = (128006, 882, 128007)
                    modules[-1]._user_header_len, modules[-1]._min_header_len = 3, 3

        if torch.cuda.is_available():
            assert not devices_per_model == 0

            if devices_per_model < 0:
                # if devices_per_model < 0: number of workers = abs(devices_per_model),
                # and devices per model is inferred based on available device count
                assert torch.cuda.device_count() % -devices_per_model == 0
                devices_per_model = int(torch.cuda.device_count() / -devices_per_model)
            else:
                # otherwise, number of workers is inferred based on available device count
                assert torch.cuda.device_count() % devices_per_model == 0

            devices = [  # splitting up available devices between worker models
                list(range(i * devices_per_model, (i + 1) * devices_per_model))
                for i in range(int(torch.cuda.device_count() / devices_per_model))
            ]
            torch_cuda_device_count = torch.cuda.device_count
            torch.cuda.device_count = lambda: len(devices[0])

            try:
                model0 = load_model('auto')
                dev_map_master = model0.hf_device_map  # template device map
                add_module(model0, devices[0][0])
            finally:  # just to be safe
                torch.cuda.device_count = torch_cuda_device_count

            for dev_idxs in devices[1:]:
                device_map = {k: dev_idxs[v] for k, v in dev_map_master.items()}  # replace template w/ correct dev. ids
                model = load_model(device_map)
                add_module(model, dev_idxs[0])
        else:
            assert devices_per_model == 1
            add_module(load_model({'': 'cpu'}), 'cpu')  # why would you use CPU????

        return cls(modules, config, _fpt=True)


class ParallelModule(torch.nn.Module):  # individual worker model
    def __init__(
            self,
            model: GRPOAgent,
            device: Union[int, str],  # input device (model may be on multiple---see ParallelModel.from_pretrained())
            teacher_model: str,
            teacher_kwargs: Dict[str, Any],
            config: "GRPOConfig",
            is_peft_model: bool  # i.e. has LoRA adapters
    ):
        super().__init__()
        self.config, self.mode, self._forward_fn = config, '', self._forward_generate
        self.model, self.device, self.is_peft_model, self._is_main = model, device, is_peft_model, False
        self.model.set_generation_kwargs(**self.config.generation_kwargs)
        self.set_mode('gen')

        # loading teacher model (copied from playpen code)
        try:
            teacher_model = teacher_model.replace("'", "\"")  # make this a proper json
            teacher_spec = ModelSpec.from_dict(json.loads(teacher_model))
        except:
            teacher_spec = ModelSpec.from_name(teacher_model)

        teacher = get_model_for(teacher_spec)
        teacher.set_gen_args(
            temperature=teacher_kwargs.pop('temperature', 0.0),  # these need to be specified
            max_tokens=teacher_kwargs.pop('max_tokens', 100),
            **teacher_kwargs
        )
        self.teacher = ClembenchAgent(model=teacher)

        train_benchmarks = {  # instantiating train games (see playpen_interface.py)
            'taboo': GameContainer(
                'taboo',
                instances=['instances_v0.9.json', 'instances_v1.0.json'],
                order=('teacher', 'agent'),
                max_reward=1.0,
                min_reward=0.0
            ),
            'wordle': GameContainer(
                'wordle',
                instances=['instances_v0.9.json', 'instances_v1.0.json'],
                order='agent',
                max_reward=1.0,
                min_reward=0.0
            ),
            'wordle_withclue': GameContainer(
                'wordle_withclue',
                instances=['instances_v0.9.json', 'instances_v1.0.json'],
                order='agent',
                max_reward=1.0,
                min_reward=0.0
            ),
            'wordle_withcritic': GameContainer(
                'wordle_withcritic',
                instances=['instances_v0.9.json', 'instances_v1.0.json'],
                order=('agent', 'teacher'),
                max_reward=1.0,
                min_reward=0.0
            ),
            'referencegame': GameContainer(
                'referencegame',
                instances=['instances_v0.9.json', 'instances_v1.0.json'],
                order=('teacher', 'agent'),
                max_reward=0.1,
                min_reward=0.0
            ),
            'imagegame': GameContainer(
                'imagegame',
                instances=['instances_v0.9.json', 'instances_v1.0.json'],
                order=('teacher', 'agent'),
                max_reward=100.0,
                min_reward=0.0
            )
        }

        if config.both_roles:
            for k, v in filter(lambda x: len(x[1].order) == 2, list(train_benchmarks.items())):
                train_benchmarks.update({
                    f'{k}_teacher': GameContainer(
                        k,
                        instances=['instances_v0.9.json', 'instances_v1.0.json'],
                        order=(v.order[1], v.order[0]),
                        min_reward=v.min_reward,
                        max_reward=v.max_reward,
                        reward_from_agent=False
                    )
                })
        if not config.no_privateshared:
            train_benchmarks.update({
                'privateshared': GameContainer(
                    'privateshared', instances=['instances_v0.9extended.json', 'instances_v1.0.json'], order='agent'
                )
            })

        # instantiating eval games
        self.benchmarks = {
            True: train_benchmarks,
            False: {  # eval benchmarks
                k: GameContainer(
                    v.name,
                    instances=['instances_v1.6.json'],
                    order=v.order,
                    max_reward=v.max_reward,
                    min_reward=v.min_reward,
                    reward_from_agent=v._rwd_from_agent
                )
                for k, v in train_benchmarks.items() if not k == 'referencegame'
            }
        }
        self.benchmarks[False].update({
            'referencegame': GameContainer(
                'referencegame',
                instances=['instances_v1.6_en.json'],  # diff. instances file name
                order=train_benchmarks['referencegame'].order,
                max_reward=train_benchmarks['referencegame'].max_reward,
                min_reward=train_benchmarks['referencegame'].min_reward
            )
        })

        # detecting model chat template tokens (for turn detection: see ParallelModule._forward_generate())
        prompt_tokens = model.tokenizer.apply_chat_template(
            [{"role": "user", "content": "playpen"}],
            return_tensors='pt',
            add_generation_prompt=True
        ).flatten().tolist()
        prompt_text = model.tokenizer.decode(prompt_tokens)
        prompt_text_split = prompt_text.split(model.tokenizer.eos_token)
        user_seq = next(iter(x for x in prompt_text_split if 'playpen' in x))
        user_seq = user_seq.replace('playpen', '').strip()
        user_toks = model.tokenizer(user_seq, return_tensors='pt')['input_ids'].flatten().tolist()
        assistant_toks = model.tokenizer(prompt_text_split[-1].strip(), return_tensors='pt')
        assistant_toks = assistant_toks['input_ids'].flatten().tolist()

        if user_toks[0] == getattr(model.tokenizer, 'bos_token_id', None):
            user_toks, assistant_toks = user_toks[1:], assistant_toks[1:]

        self._user_header, self._assistant_header = tuple(user_toks), tuple(assistant_toks)
        self._user_header_len, self._assistant_header_len = len(self._user_header), len(self._assistant_header)
        self._min_header_len = min(self._user_header_len, self._assistant_header_len)
        self._header_tok = self.model.tokenizer.eos_token_id

    def instance_idxs(self, train: bool = True) -> List[Tuple[str, int]]:
        return list(chain.from_iterable(  # returns (game_name, index) pairs for train/test set
            ((k, i) for i in range(len(v))) for k, v in self.benchmarks[train].items()
        ))

    def set_mode(self, mode: str) -> None:
        # sets model up for relevant job (for self._forward_fn, see ParallelModule.forward())
        if not mode == self.mode:
            assert mode in {'train', 'gen', 'eval', 'fwd'}
            self.mode = mode

            if self.mode == 'train':
                assert self._is_main
                self.model.model.train()
                self._forward_fn = self._forward_train
            else:
                self.model.model.eval()
                self._forward_fn = self._forward_ref if self.mode == 'fwd' else self._forward_generate

                if mode == 'eval':
                    self.model.set_generation_kwargs(do_sample=False, temperature=None, top_k=None, top_p=None)
                else:
                    self.model.set_generation_kwargs(**self.config.generation_kwargs)

    def forward(
            self,
            *args
    ) -> Union[
        Dict[Tuple[str, int], List[_generate_output_type]],
        List[torch.Tensor], Generator[torch.Tensor, None, None]
    ]:
        # torch parallel_apply calls model's forward() function: change the forward function based on mode (generate
        # for eval/generate, or actual forward pass for train/ref_logits)
        return self._forward_fn(*args)

    def _forward_ref(  # forward function for generating reference logits
            self,
            inputs: Dict[str, torch.Tensor],
            n_logits_to_keep: int,
            progress_bar: bool
    ) -> List[torch.Tensor]:
        # if model has LoRA, just disable adapters to compute ref. logits
        with (self.model.model.disable_adapter() if self.is_peft_model else null_context()) as _:
            return list(_forward_pass(
                self.model.model,
                inputs,
                self.device,
                n_logits_to_keep,
                self.config.grad_acc_batch_size,
                progress_bar=(self._is_main and progress_bar),
                desc=('Computing reference logits' if self._is_main and progress_bar else None)
            ))

    def _forward_generate(  # forward function for generating train/eval trajectories
            self,
            game_idxs: List[Tuple[str, int]],
            progress_bar: bool,
            train: bool
    ) -> Dict[Tuple[str, int], List[_generate_output_type]]:
        out_dict = {}
        desc = 'Generating trajectories' if train else 'Evaluating'
        mb_iter = tqdm_iter(  # just for tqdm progress bar
            len(game_idxs),
            progress_bar=(self._is_main and progress_bar),
            desc=(desc if self._is_main and progress_bar else None)
        )

        with (torch.no_grad() as _, set_batch_size(self.model.model, 1) as _):  # main loop
            for mb in mb_iter:
                game, idx = game_idxs[mb]
                response, mask, reward, query, dialog = None, None, None, None, None

                for _ in range(self.config.retry_limit):  # retries for teacher-aborted episodes
                    try:
                        agent_obs, reward = self.benchmarks[train][game].play(
                            idx, self.model, self.teacher,
                            pos_reward=self.config.pos_reward,
                            neg_reward=self.config.neg_reward,
                            fail_reward=self.config.fail_reward
                        )
                    except RuntimeError:  # catches NaN in probability tensor (happens sometimes w/ SFT'd 4bit models)
                        agent_obs = ()
                        warnings.warn('NaN in probability tensor')

                    if len(agent_obs) > 0:
                        response = self.model.tokenizer.apply_chat_template(
                            agent_obs, add_generation_prompt=False, return_tensors='pt'
                        ).flatten()

                    if reward is not None:  # teacher didn't abort
                        reward = torch.tensor(reward)
                        turns, mask_list, last_i, past_header, in_query = [], [], 0, False, False
                        response_list = response.tolist()

                        for i in range(len(response_list) - self._min_header_len):  # computes turns & masks
                            if past_header:  # header = tokenizer-generated dialog header
                                if tuple(response_list[i: i + self._assistant_header_len]) == self._assistant_header:
                                    # end non-agent turn
                                    turns.append(response_list[last_i:i + self._assistant_header_len])

                                    if in_query:  # (query = tokenizer-gen'd dialog header + first non-agent turn)
                                        in_query = False  # don't need query for weight update, so we ignore it
                                    else:
                                        mask_list.extend(0 for _ in range(last_i, i + self._assistant_header_len))

                                    last_i = i + self._assistant_header_len
                                elif tuple(response_list[i: i + self._user_header_len]) == self._user_header:
                                    # end agent turn
                                    turns.append(response_list[last_i:i])
                                    mask_list.extend(1 for _ in range(last_i, i))
                                    last_i = i
                            elif past_header is None:  # start detecting turns one token after end of header
                                past_header = True
                            elif response_list[i] == self._header_tok:  # detect end of header
                                past_header = None

                        turns.append(response_list[last_i:])  # last turn not detected in above loop
                        mask_list.extend(1 for _ in range(last_i, len(response_list)))

                        query = f'({game}, {idx}) {self.model.tokenizer.decode(turns[0])}'
                        dialog = ''.join(map(self.model.tokenizer.decode, turns[1:]))
                        mask = torch.tensor(mask_list)

                        break
                else:  # teacher model aborted retry_limit times in a row
                    query = f'({game}, {idx}: ABORTED) '
                    dialog = '' if response is None else self.model.tokenizer.decode(response)
                    response, mask, reward = None, None, None
                if (game, idx) in out_dict.keys():  # out_dict aggregates trajectories by group (i.e. instance)
                    out_dict[(game, idx)].append((response, mask, reward, query, dialog))
                else:
                    out_dict.update({(game, idx): [(response, mask, reward, query, dialog)]})

        return out_dict

    def _forward_train(  # forward function for updating weights of main worker model
            self,
            inputs: Dict[str, torch.Tensor],
            n_logits_to_keep: int
    ) -> Generator[torch.Tensor, None, None]:
        yield from _forward_pass(
            self.model.model, inputs, self.device, n_logits_to_keep, self.config.grad_acc_batch_size
        )


def _forward_pass(  # shared by ParallelModule._forward_train() and ParallelModule._forward_ref()
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        device: Union[str, int],
        n_logits_to_keep: int,  # cuts off header
        grad_acc_batch_size: int,
        progress_bar: bool = False,
        desc: Optional[str] = None
) -> Generator[torch.Tensor, None, None]:
    for n in tqdm_iter(-(-inputs['input_ids'].size(0) // grad_acc_batch_size), progress_bar=progress_bar, desc=desc):
        batch_inputs = {
            k: v[n * grad_acc_batch_size:(n + 1) * grad_acc_batch_size].to(device) for k, v in inputs.items()
        }

        with set_batch_size(model, batch_inputs['input_ids'].size(0)):
            # last logit = next predicted token, so we ignore it
            logits = model(**batch_inputs, num_logits_to_keep=(n_logits_to_keep + 1)).logits[:, :-1, :]

        yield torch.stack(tuple(  # yields policy log probs for all tokens in input trajectory
            torch.gather(logits_row.log_softmax(dim=-1), dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            for logits_row, input_ids_row in zip(logits, batch_inputs['input_ids'][:, -n_logits_to_keep:])
        ))
