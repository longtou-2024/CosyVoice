# lontou.2024
from pathlib import Path
import io
import json

import webdataset as wds

#name2url = {
#    "azure": "gs://ai-lab-speech-bucket/longtou/db/azure/wds_v2/shard-00000{0..7}.tar",
#    "mediazen": "gs://ai-lab-speech-bucket/longtou/db/mediazen/wds_v2/shard-00{0000..1055}.tar",
#    "literature": "gs://ai-lab-speech-bucket/longtou/db/literature/wds_v2/shard-0000{00..46}.tar",
#}
name2url = {
    "azure": "gs://ai-lab-speech-bucket/longtou/db/azure/wds_v2/shard-00000{0,1}.tar",
    "mediazen": "gs://ai-lab-speech-bucket/longtou/db/mediazen/wds_v2/shard-00000{0,1}.tar",
    "literature": "gs://ai-lab-speech-bucket/longtou/db/literature/wds_v2/shard-00000{0,1}.tar",
}

### implement decoding function for each dataset ###

def decode_azure(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"]
    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_mediazen(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["전사정보"]["OrgLabelText"]
    return {"utt": uttid, "audio_data": wav, "text": transcript}

def decode_literature(sample):
    uttid = sample["__key__"]
    wav = sample["wav"]
    json_data = json.load(io.BytesIO(sample["json"]))
    transcript = json_data["transcript"]
    return {"utt": uttid, "audio_data": wav, "text": transcript}

### end of decoding function list ###


def build_wds(recipe_name):
    shard_url = name2url[recipe_name]
    cache_dir = f"wds_cache_{recipe_name}"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _decode = globals()[f"decode_{recipe_name}"]
    dataset = wds.WebDataset(
        shard_url,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        cache_dir=cache_dir,
        cache_size=int(3e9),
        shardshuffle=100,
        resampled=False,
        repeat=False,
    )

    dataset = dataset.map(_decode)
    return dataset

