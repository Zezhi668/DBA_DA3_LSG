import torch
from datetime import datetime
import yaml

try:
    import pytz
except ModuleNotFoundError:
    pytz = None

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:
    ZoneInfo = None

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_name(cfg=None):
    if pytz is not None:
        beijing_tz = pytz.timezone('Asia/Shanghai')
    elif ZoneInfo is not None:
        beijing_tz = ZoneInfo('Asia/Shanghai')
    else:
        beijing_tz = None
    now = datetime.now(beijing_tz)
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    current_minute = now.minute
    if cfg is not None:
        formatted_string = f"{current_month:02d}-{current_day:02d}-{current_hour:02d}-{current_minute:02d}-{cfg['dataset']['module'].split('.')[-1]}"
    else:
        formatted_string = f"{current_month:02d}-{current_day:02d}-{current_hour:02d}-{current_minute:02d}"
    return formatted_string

def load_config(cfg_path):
    # Return a Dict.
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.full_load(f)
    return cfg 
