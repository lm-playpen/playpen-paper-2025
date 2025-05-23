
import re
import os
import json
import warnings
import argparse
from datetime import datetime
from typing import Optional, Any, Dict, Tuple, Set, Union
from training.utils import get_path, check_fn_ow, chkpt_name


# parses command-line arguments for train.py: this was mostly useful for the other project that I copy/pasted
# the majority of this code from, and probably could have just been handled with argparse in this project...
# (feel free to mostly ignore)


_parse_args_return_type = Tuple[
    Dict[str, Union[int, float, str, bool]],
    Dict[str, Union[
        Dict[str, Union[int, float, str, bool]],
        Dict[str, dict],
        Dict[str, Union[list, int]],
        str
    ]],
    str,
    Optional[str],
    bool,
    Optional[int]
]


class CmdArg:  # single command-line argument
    def __init__(
            self,
            name: str,
            flag: Optional[str] = None,
            default_val: Any = None,
            is_default: Optional[bool] = False,
            **kwargs
    ):
        assert flag is None or len(flag) == 1
        self.name, self.flag, self.kwargs = name, flag, kwargs
        self.default_val, self.is_default = default_val, is_default
        self.store_true = self.kwargs.get('action', None) == 'store_true'
        self.is_default = is_default is True or default_val is not None
        self.default_val = default_val

        assert not (default_val is not None and self.is_default is False)
        assert not (self.store_true and (is_default or default_val is not None))

        if 'help' in self.kwargs.keys():
            helpstr = self.kwargs['help'].replace('%%%', str(self.default_val)).replace('\n', ' ').replace('\t', ' ')

            while '  ' in helpstr:
                helpstr = helpstr.replace('  ', ' ')

            self.kwargs['help'] = helpstr.strip()

    def add_to_argparse(self, ap: argparse.ArgumentParser) -> None:
        if self.flag is None:
            ap.add_argument(f'--{self.name}', **self.kwargs)
        else:
            ap.add_argument(f'-{self.flag}', f'--{self.name}', **self.kwargs)

    def _as_str(self, tab: int = 0) -> str:
        out_str, tab_str = f'--{self.name}', (' ' * 4)

        if self.flag is not None:
            out_str = f'{self.flag} ({out_str})'

        out_str = f'{tab_str * tab}{out_str}: {self.kwargs.get("help", None)}\n'
        out_str += '\n'.join(f'{tab_str * (tab + 1)}{k}: {v}' for k, v in self.kwargs.items() if not k == 'help')

        return out_str.rstrip()

    def __str__(self):
        return self._as_str()


class CmdArgSet:  # acts like argparse (with some post-processing)
    def __init__(self, *cmd_args: "CmdArg"):
        self.cmd_args, self.default_args, self.store_true, self._names, self._flags = [], {}, set(), set(), set()
        self.add_args(*cmd_args)

    def add_args(self, *cmd_arg: "CmdArg") -> None:
        for c in cmd_arg:
            assert c.name not in self._names and c.flag not in self._flags, f'{c.name}, {self._names}'
            self.cmd_args.append(c)
            self._names.add(c.name)

            if c.flag is not None:
                self._flags.add(c.flag)
            if c.is_default:
                self.default_args.update({c.name: c.default_val})
            elif c.store_true:
                self.store_true.add(c.name)

    def merge(self, cmd_arg_set: "CmdArgSet") -> None:
        self.add_args(*cmd_arg_set.cmd_args)

    def parse_args(self, filepath: str, *args) -> _parse_args_return_type:  # acts like argparse.parse_args()
        ap = argparse.ArgumentParser()

        for cmdarg in self.cmd_args:
            cmdarg.add_to_argparse(ap)

        ap_pa = ap.parse_args(*args)
        ap_args = {
            k: v for k, v in ((k_, getattr(ap_pa, k_)) for k_ in dir(ap_pa)) if not
            (k[0] == '_' or v is None or (k in self.store_true and not v))
        }

        lora = any(k.startswith('lora') for k in ap_args.keys())
        load_from_run = ap_args.pop('load_from_run', None)
        run_name = ap_args.pop('run_name', None)
        stat_fp, _ = get_path(ap_args.pop('stat_fp', None), 'stats', realpath=filepath)
        dpm = ap_args.pop('device_per_model', 1)

        if run_name is None:  # create default run name
            max_run = max(
                (int(x.split('.')[0][4:]) for x in os.listdir(stat_fp) if re.match(r'run_\d+\.[^.]+$', x)),
                default=-1
            )
            run_name = f'run_{max_run + 1}.json'

        rn0, rn1 = run_name.rsplit('.', maxsplit=1) if '.' in run_name else (run_name, '')
        save_model = ap_args.pop('save_model', False)

        if load_from_run is None:  # not loading from checkpoint
            save_fp = check_fn_ow(stat_fp, rn0, ('.' * ('.' in run_name)) + rn1)  # make sure we're not overwriting
            test_res, prev_time, load_chkpt = [], '', None
            load_from_run_file = {
                'ep_data': {},
                'test': {'interval': ap_args['eval_interval'], 'steps': []}  # TODO: keyerror
            }

            with open(save_fp, 'w') as f:  # prevent name clash when running multiple jobs
                f.write('')
        else:  # loading from checkpoint
            load_fp = os.path.abspath(load_from_run) if '/' in load_from_run else f'{stat_fp}{load_from_run}'
            load_chkpt = chkpt_name(load_fp)
            save_fp = load_fp

            with open(load_fp, 'r') as f:
                load_from_run_file = json.load(f)

            ap_args = {**load_from_run_file['args'], **ap_args}  # use hyperparams from prev. run as defaults
            lora = lora or load_from_run_file['args'].get('lora', False)

            if not lora == load_from_run_file['args'].get('lora', False):
                raise ValueError('LoRA')

            for k, v in ap_args.items():
                if k.startswith('lora'):  # and lora and not k == 'lora':
                    if lora and (not k == 'lora') and not v == load_from_run_file['args'][k]:
                        raise ValueError(k)
                elif k in load_from_run_file['args'].keys() and not v == load_from_run_file['args'][k]:
                    raise ValueError(k)

            prev_time, test_res = f'{load_from_run_file["time"]}, ', []
        if not save_model:  # model checkpointing
            warnings.warn('\'--save_model\' flag not used---model will not be checkpointed', UserWarning)

        ap_args.update({k: False for k in self.store_true - set(ap_args.keys())})  # set remaining store_true vals
        ap_args = {**self.default_args, **ap_args}  # set remaining default vals
        save_file = {
            'args': dict(ap_args),  # copy ap_args for save_file (mostly relevant for other project)
            'ep_data': load_from_run_file['ep_data'],
            'time': prev_time + str(datetime.now()),
            'test': load_from_run_file['test']
        }

        if lora:
            save_file['args'].update({'lora': True})
            ap_args.update({'lora': True})
        else:
            for k in (k_ for k_ in ap_args.keys() if k_.startswith('lora')):
                save_file['args'].pop(k)

        return ap_args, save_file, save_fp, load_chkpt, save_model, dpm

    def defaults(self) -> Tuple[Dict[str, Any], Set[str]]:
        return self.default_args, self.store_true


grpo_arg_set = CmdArgSet(*[  # command-line args for train.py
    CmdArg(
        'stat_fp',
        flag='f',
        type=str,
        help="""
    Path to directory used for storing stats/data and model checkpoints. Default: 
    {path to project}/stats/ (creates this directory if it doesn\'t exist)
    """
    ),
    CmdArg(
        'run_name',
        flag='n',
        type=str,
        help="""
    Name of the current run: used to save stats/data in the {data_fp} directory. 
    Default: run_{number of existing runs + 1}.json
    """
    ),
    CmdArg(
        'save_model',
        flag='s',
        action='store_true',
        help='Checkpoint the model (saved after each eval step to \"{data_fp}/{run_name}/\")'
    ),
    CmdArg(
        'load_from_run',
        flag='r',
        type=str,
        help="""
    Path to previous run to resume: uses the args from this run as defaults for the current run (can be overridden). 
    Default: None
    """
    ),
    CmdArg(
        'model_id',
        flag='i',
        type=str,
        default_val='meta-llama/Meta-Llama-3.1-8B-Instruct',
        help='Huggingface model id. Default: %%%'
    ),
    CmdArg(
        'access_token',
        flag='a',
        type=str,
        is_default=True,
        help='Huggingface access token.'
    ),
    CmdArg(
        'batch_size',
        flag='b',
        type=int,
        default_val=16,
        help='Step batch size. Default: %%%'
    ),
    CmdArg(
        'group_size',
        flag='g',
        type=int,
        default_val=8,
        help='Number of parallel environments. Default: %%%'
    ),
    CmdArg(
        'learn_rate',
        flag='l',
        type=float,
        default_val=5e-7,
        help='Training learn rate. Default: %%%'
    ),
    CmdArg(
        'epochs',
        flag='e',
        type=int,
        default_val=1,
        help='Number of training epochs. Default: %%%'
    ),
    CmdArg(
        'eval_interval',
        type=int,
        default_val=0,
        help='Number of training steps between evaluation steps. Default: %%%'
    ),
    CmdArg(
        'device_per_model',
        type=int,
        help='Devices per model for parallel generation. Default: 1'
    ),
    CmdArg(
        'initial_eval',
        action='store_true',
        help='Perform intial evaluation.'
    ),
    CmdArg(
        'teacher_model',
        type=str,
        default_val='gpt-4o-mini-2024-07-18',
        help='Teacher model name. Default: %%%'
    ),
    CmdArg(
        'teacher_temperature',
        type=float,
        default_val=0.0,
        help='Teacher model temperature. Default: %%%'
    ),
    CmdArg(
        'retry_limit',
        type=int,
        default_val=1,
        help='Number of times to retry when the episode is aborted due to teacher model error. Default: %%%'
    ),
    CmdArg(
        'teacher_max_tokens',
        type=int,
        default_val=100,
        help='Teacher model max generated tokens. Default: %%%'
    ),
    CmdArg(
        'grad_acc_batch_size',
        type=int,
        default_val=1,
        help='Size of gradient accumulation step(s). Default: %%%'
    ),
    CmdArg(
        'no_ps',
        action='store_true',
        help='Remove privateshared from training/evaluation data.'
    ),
    CmdArg(
        'both_roles',
        action='store_true',
        help='Include games where the agent plays as teacher.'
    ),
    CmdArg(
        'no_mask',
        action='store_true',
        help='Disable TRL environment response masking.'
    ),
    CmdArg(
        'lora',
        action='store_true',
        help='Enable LoRA for the model.'
    ),
    CmdArg(
        'lora_r',
        default_val=16,
        type=int,
        help='LoRA \'r\' kwarg (filling this argument automatically enables LoRA). Default: %%%'
    ),
    CmdArg(
        'lora_alpha',
        default_val=32,
        type=int,
        help='LoRA \'alpha\' kwarg (filling this argument automatically enables LoRA). Default: %%%'
    ),
    CmdArg(
        'lora_dropout',
        default_val=0.05,
        type=float,
        help='LoRA \'dropout\' kwarg (filling this argument automatically enables LoRA). Default: %%%'
    ),
    CmdArg(
        'lora_layers',
        default_val='all',
        type=str,
        help='Model layers to target with LoRA (\'all\', \'qv\', or \'qvko\'). Default: \'%%%\''
    ),
    CmdArg(
        'pos_reward',
        default_val=1.0,
        type=float,
        help='Positive reward value. Default: %%%'
    ),
    CmdArg(
        'neg_reward',
        default_val=0.0,
        type=float,
        help='Positive reward value. Default: %%%'
    ),
    CmdArg(
        'fail_reward',
        default_val=0.0,
        type=float,
        help='Failure (i.e. format) reward value. Default: %%%'
    ),
    CmdArg(
        'seed',
        is_default=True,
        type=int,
        help='Manual seed for random and pytorch modules. Default: None (random seed)'
    ),
    CmdArg(
        'top_k',
        default_val=0,
        type=int,
        help='\'top_k\' kwarg in model\'s generation config. Default: %%%'
    ),
    CmdArg(
        'top_p',
        default_val=1.0,
        type=float,
        help='\'top_p\' kwarg in model\'s generation config. Default: %%%'
    ),
    CmdArg(
        'temperature',
        default_val=1.0,
        type=float,
        help='\'temperature\' kwarg in model\'s generation config. Default: %%%'
    ),
    CmdArg(
        'max_new_tokens',
        default_val=100,
        type=int,
        help='\'max_new_tokens\' kwarg in model\'s generation config. Default: %%%'
    ),
    CmdArg(
        'beta',
        default_val=0.04,
        type=float,
        help='GRPO beta hyperparameter. Default: %%%'
    )
])

if __name__ == '__main__':  # for printing args for debugging
    print('\n'.join(a._as_str(tab=1) for a in grpo_arg_set.cmd_args))
