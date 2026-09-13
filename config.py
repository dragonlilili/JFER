# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


from pathlib import Path

import yaml
from easydict import EasyDict

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def _resolve_base_config_path(base_config, current_dir=None):
    base_path = Path(base_config)
    candidates = [base_path]
    if current_dir is not None and not base_path.is_absolute():
        candidates.append(Path(current_dir) / base_path)
    if not base_path.is_absolute():
        repo_root = (Path(__file__).resolve().parent / '../').resolve()
        candidates.append(repo_root / 'tools' / base_path)
        candidates.append(repo_root / base_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_path


def merge_new_config(config, new_config, current_dir=None):
    if '_BASE_CONFIG_' in new_config:
        base_config_path = _resolve_base_config_path(
            new_config['_BASE_CONFIG_'], current_dir=current_dir
        )
        with open(base_config_path, 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        merge_new_config(
            config=config,
            new_config=EasyDict(yaml_config),
            current_dir=base_config_path.parent,
        )

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, current_dir=current_dir)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config, current_dir=Path(cfg_file).parent)

    return config



cfg = EasyDict()
cfg.ROOT_DIR = Path(__file__).resolve().parent
cfg.LOCAL_RANK = 0
