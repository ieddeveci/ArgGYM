from pathlib import Path
import json

import datasets


MANIFEST = Path(
    "/arf/scratch/futan/ArgGYM/"
    "outputs/transfer/protocol_freeze/"
    "lmeval_datasets_20260920.json"
)

m = json.loads(
    MANIFEST.read_text()
)

assert m["record_count"] == 29
assert m["bbh_config_count"] == 27


for i, expected in enumerate(
    m["datasets"],
    start=1,
):
    path = expected["dataset_path"]
    config = expected["config"]
    revision = expected["revision"]

    print(
        f"OFFLINE {i:02d}/29: "
        f"{path} [{config}]",
        flush=True,
    )

    if config is None:
        ds = datasets.load_dataset(
            path,
            revision=revision,
        )
    else:
        ds = datasets.load_dataset(
            path,
            config,
            revision=revision,
        )

    assert set(ds.keys()) == set(
        expected["splits"].keys()
    )

    for split_name, exp in (
        expected["splits"].items()
    ):
        split = ds[split_name]

        assert len(split) == exp["rows"]

        assert (
            split._fingerprint
            == exp["fingerprint"]
        )

        assert (
            list(split.column_names)
            == exp["columns"]
        )


print(
    "LM-EVAL DATASET OFFLINE RELOAD: PASS"
)
