import json
import hashlib
from pathlib import Path


ROOT = Path("experiments")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_json(obj):
    raw = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


dev = load(
    ROOT / "frozen/researchy_dev20/datachunk_0.json"
)

rule_pool = load(
    ROOT
    / "frozen/researchy_rulepool100/datachunk_0.json"
)

test = load(
    ROOT / "frozen/researchy_test50/datachunk_0.json"
)

quality = load(
    ROOT / "splits/researchy_quality10_ids.json"
)

metadata = load(
    ROOT
    / "config/researchy_split_metadata.json"
)


dev_ids = set(dev)
rule_ids = set(rule_pool)
test_ids = set(test)
quality_ids = set(quality)


assert len(dev) == 20
assert len(rule_pool) == 100
assert len(test) == 50
assert len(quality) == 10


assert dev_ids.isdisjoint(rule_ids)
assert dev_ids.isdisjoint(test_ids)
assert rule_ids.isdisjoint(test_ids)

assert quality_ids.issubset(test_ids)


for name, dataset in [
    ("dev", dev),
    ("rule_pool", rule_pool),
    ("test", test),
]:

    for qid, item in dataset.items():

        assert "query" in item
        assert "text_list" in item
        assert "target_id" in item

        assert len(item["text_list"]) == 5

        assert (
            0
            <= item["target_id"]
            < 5
        )


test_hash = sha256_json(test)

expected_hash = (
    metadata["hashes"]["test50_sha256"]
)

assert test_hash == expected_hash


print("All checks passed.")
print()
print("Dev20       :", len(dev))
print("RulePool100 :", len(rule_pool))
print("Test50      :", len(test))
print("Quality10   :", len(quality))
print()
print("Test hash:")
print(test_hash)