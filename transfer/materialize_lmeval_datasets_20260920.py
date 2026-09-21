from pathlib import Path
import hashlib
import importlib.metadata
import json

import datasets


ROOT = Path("/arf/scratch/futan/ArgGYM")

PIN = (
    ROOT
    / "transfer/lm_eval_tasks/pinned_20260920"
)

CACHE = Path(
    "/arf/scratch/futan/"
    "arggym_transfer_runtime/"
    "hf_transfer_cache_20260920"
)

OUT = (
    ROOT
    / "outputs/transfer/protocol_freeze/"
    "lmeval_datasets_20260920.json"
)

REVISIONS = {
    "gsm8k":
        "740312add88f781978c0658806c59bc2815b9866",
    "mmlu_pro":
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "bbh":
        "b5306be6f827cfafbb545ff5a51f96916029b0fd",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def file_record(filename: str):
    p = Path(filename).resolve()
    cache_root = CACHE.resolve()

    if not p.is_relative_to(cache_root):
        raise RuntimeError(
            f"Dataset cache file escaped dedicated cache: {p}"
        )

    return {
        "path":
            p.relative_to(cache_root).as_posix(),
        "bytes":
            p.stat().st_size,
        "sha256":
            sha256(p),
    }


def dataset_record(
    dataset_path: str,
    config,
    revision: str,
    ds,
):
    splits = {}

    for split_name in sorted(ds.keys()):
        split = ds[split_name]

        filenames = sorted(
            {
                item["filename"]
                for item in split.cache_files
                if "filename" in item
            }
        )

        splits[split_name] = {
            "rows": len(split),
            "fingerprint":
                split._fingerprint,
            "columns":
                list(split.column_names),
            "cache_files": [
                file_record(x)
                for x in filenames
            ],
        }

    return {
        "dataset_path": dataset_path,
        "config": config,
        "revision": revision,
        "splits": splits,
    }


# Read exact BBH dataset configurations from the
# already-frozen task definitions.
bbh_configs = []

for p in sorted(
    (PIN / "bbh").glob("*.yaml")
):
    if p.name.startswith("_"):
        continue

    config = None

    for line in p.read_text().splitlines():
        if line.startswith("dataset_name:"):
            config = (
                line.split(":", 1)[1]
                .strip()
                .strip("\"'")
            )
            break

    if config is None:
        raise RuntimeError(
            f"No dataset_name in {p}"
        )

    bbh_configs.append(config)


assert len(bbh_configs) == 27
assert len(set(bbh_configs)) == 27

records = []


print("PREFETCH GSM8K", flush=True)

gsm = datasets.load_dataset(
    "openai/gsm8k",
    "main",
    revision=REVISIONS["gsm8k"],
)

records.append(
    dataset_record(
        "openai/gsm8k",
        "main",
        REVISIONS["gsm8k"],
        gsm,
    )
)


print("PREFETCH MMLU-PRO", flush=True)

mmlu = datasets.load_dataset(
    "TIGER-Lab/MMLU-Pro",
    revision=REVISIONS["mmlu_pro"],
)

records.append(
    dataset_record(
        "TIGER-Lab/MMLU-Pro",
        None,
        REVISIONS["mmlu_pro"],
        mmlu,
    )
)


for i, config in enumerate(
    bbh_configs,
    start=1,
):
    print(
        f"PREFETCH BBH {i:02d}/27: {config}",
        flush=True,
    )

    ds = datasets.load_dataset(
        "SaylorTwift/bbh",
        config,
        revision=REVISIONS["bbh"],
    )

    records.append(
        dataset_record(
            "SaylorTwift/bbh",
            config,
            REVISIONS["bbh"],
            ds,
        )
    )


assert len(records) == 29

manifest = {
    "status":
        "LM_EVAL_DATASETS_MATERIALIZED_PRE_EVAL",

    "runtime": {
        "datasets":
            datasets.__version__,
        "huggingface_hub":
            importlib.metadata.version(
                "huggingface-hub"
            ),
    },

    "cache_root":
        str(CACHE),

    "dataset_revisions":
        REVISIONS,

    "bbh_config_count":
        27,

    "record_count":
        len(records),

    "datasets":
        records,
}


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if OUT.exists():
    raise RuntimeError(
        f"Refusing to overwrite {OUT}"
    )

OUT.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print(
    "LM-EVAL DATASET MATERIALIZATION: PASS",
    flush=True,
)
print(
    "records:",
    len(records),
    flush=True,
)
print(
    "manifest_sha256:",
    sha256(OUT),
    flush=True,
)
