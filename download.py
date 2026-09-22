import json
from pathlib import Path
from huggingface_hub import snapshot_download
ROOT = Path(__file__).resolve().parent
REVISION = '1c5edc17a7acd8701df6fc341c0d179f1c62c982'
patterns = ['rl_agent_config.json', 'model.safetensors', 'tokenizer/*', 'encoder/*', 'multilingual/*']
snapshot_download('convaiinnovations/laya', revision=REVISION, allow_patterns=patterns, local_dir=ROOT / 'models', max_workers=2)
(ROOT / 'models' / 'revision.txt').write_text(REVISION + '\n')
print('Download complete:', REVISION)
