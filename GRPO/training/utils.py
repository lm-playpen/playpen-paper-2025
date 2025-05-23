
import os
from tqdm import tqdm
from collections import OrderedDict
from contextlib import contextmanager
from transformers import PreTrainedModel
from peft import load_peft_weights, get_peft_model_state_dict
from typing import TYPE_CHECKING, Optional, Generator, Iterator, TypeVar, List, Tuple, Union


if TYPE_CHECKING:  # :eyeroll emoji:
    from training.trainer import GRPOTrainer
    from .parallelization import ParallelModule
else:
    GRPOTrainer, ParallelModule = None, None

_any_type = TypeVar('_any_type', bound="Any")
LORA_LAYER_MAP = {
    'all': 'all-linear',
    'qv': None,
    'qvko': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
}


@contextmanager
def set_batch_size(
        model: PreTrainedModel,
        batch_size: int,
        next_batch_size: Optional[int] = None
) -> Generator[None, None, None]:
    try:
        if hasattr(model, 'config') and hasattr(model.config, 'batch_size'):
            next_batch_size = model.config.batch_size if next_batch_size is None else next_batch_size
            model.config.batch_size = batch_size
        else:
            next_batch_size = None

        yield
    finally:
        if next_batch_size is not None:
            model.config.batch_size = next_batch_size


@contextmanager
def null_context() -> Generator[None, None, None]:
    try:
        yield
    finally:
        pass


def tqdm_iter(
        n_steps: int,
        generator: Optional[Iterator[_any_type]] = None,
        progress_bar: bool = True,
        **tqdm_kwargs
) -> Generator[_any_type, None, None]:
    generator = iter(range(n_steps)) if generator is None else generator

    if progress_bar:
        for _ in tqdm(range(n_steps), **tqdm_kwargs):
            yield next(generator)
    else:
        yield from generator


def eval_model(
        trainer: "GRPOTrainer",
        save_file: dict,
        test_rewards: List[float],
        test_accs: List[float],
        header: str = ''
) -> None:
    print(f'{(hs + " ") if len(hs := header.strip()) > 0 else ""}EVAL:\n\n')

    eval_v = trainer.eval()
    save_file['test']['steps'].append(eval_v)
    test_rewards.append(round(eval_v['mean_reward'], 3))
    test_accs.append(round(eval_v['accuracy'], 3))

    for game, game_stats in eval_v['games'].items():
        print(
            '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'.join(
                f'({game}, {ev_idx}) {ev_trj}' for ev_idx, ev_trj in game_stats['responses'].items()
            )
        )

    print('\n\n')

    for game, game_stats in eval_v['games'].items():
        print(f'REWARD ({game}): {game_stats["mean_reward"]}\nACC ({game}): {game_stats["accuracy"]}')

    print(f'REWARD (overall): {eval_v["mean_reward"]}\nACC (overall): {eval_v["accuracy"]}\n\n\n')


def get_path(
        filepath: Optional[str],
        default_name: str,
        mkdir: bool = False,
        realpath: Optional[str] = None
) -> Tuple[str, bool]:
    if filepath is None:
        fp0, _, _ = (os.path.realpath(__file__) if realpath is None else realpath).rsplit('/', maxsplit=2)
        filepath, is_default = f'{fp0}/{default_name}/', True

        if mkdir and default_name not in os.listdir(fp0):
            os.mkdir(filepath)
    else:
        filepath, is_default = os.path.abspath(filepath), False
        filepath += '' if filepath[-1] == '/' else '/'

    return filepath, is_default


def check_fn_ow(filepath: str, file_prefix: str, file_type: str) -> str:  # checks filename overwrite
    fp_fns = set(os.listdir(filepath))

    if file_prefix + file_type in fp_fns:
        sfx = 0

        while f'{file_prefix}({sfx}){file_type}' in fp_fns:
            sfx += 1

        return f'{filepath}{file_prefix}({sfx}){file_type}'

    return filepath + file_prefix + file_type


def chkpt_name(save_filepath: str) -> str:  # derives checkpoint name from data file name
    save_path, save_fn = save_filepath.rsplit('/', maxsplit=1)
    save_fn0 = save_fn.rsplit('.', maxsplit=1)[0] if '.' in save_fn else save_fn

    return f'{save_path}/{save_fn0}'


def peft_sd(module_or_path: Union["ParallelModule", str]) -> OrderedDict:
    if isinstance(module_or_path, str):
        sd = load_peft_weights(module_or_path)
    else:
        sd = get_peft_model_state_dict(module_or_path.model.model)

    return sd_to_cpu(sd)


def sd_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    sd_out = OrderedDict()

    for k, v in state_dict.items():
        sd_out.update({k: v.to('cpu')})

    return sd_out
