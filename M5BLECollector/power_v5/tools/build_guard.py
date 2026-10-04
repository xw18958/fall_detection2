"""Application-only deployment with a verified V5 data/model contract."""
Import('env')
from pathlib import Path
import sys
if any(t in {'upload','uploadfs','erase','erase_flash'} for t in COMMAND_LINE_TARGETS):
    raise RuntimeError('Generic upload/erase disabled; use the reviewed inactive-OTA-slot deployment package')
project=Path(env.subst('$PROJECT_DIR'))
sys.path.insert(0,str(project.parent/'tools'))
from verify_power_contract import verify
verify()
raw_flags=env.GetProjectOption('build_flags', [])
flags=raw_flags if isinstance(raw_flags,str) else ' '.join(raw_flags)
trigger='-DCONFIG_FALL_TRIGGER_MODEL=1' in flags
gated='-DCONFIG_FALL_CASCADE=1' in flags
config=project/('sdkconfig.'+env.subst('$PIOENV'))
if config.exists():
    text=config.read_text()
    trigger=trigger or 'CONFIG_FALL_TRIGGER_MODEL=y' in text
    gated=gated or 'CONFIG_FALL_CASCADE=y' in text
if trigger:
    from verify_trigger_bundle import verify as verify_trigger
    verify_trigger(project/'main', gated)
