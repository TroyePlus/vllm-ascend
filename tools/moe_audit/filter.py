"""Compact all-rank audit; stdlib only. Usage: python filter.py prefill.log."""

import argparse
import json
from collections import Counter, defaultdict


def summarize(paths):
    records = []
    versions = set()
    malformed = 0
    backend_entries = Counter()
    binding_failures = 0
    for path in paths:
        with open(path, errors="replace") as source:
            for line in source:
                if "Enter external FX backend graph_id=" in line:
                    backend_entries[line.split("pid=", 1)[-1].split(";", 1)[0]] += 1
                if "Bind cpus failed" in line:
                    binding_failures += 1
                if "[MOE_AUDIT_VERSION]" in line:
                    versions.add(line.split("[MOE_AUDIT_VERSION]", 1)[1].strip())
                for part in line.split("[MOE_AUDIT] ")[1:]:
                    try:
                        row = json.JSONDecoder().raw_decode(part)[0]
                        if not isinstance(row, dict) or "event" not in row:
                            raise ValueError("not an audit record")
                        records.append(row)
                    except (ValueError, TypeError):
                        malformed += 1
    for version in sorted(versions):
        print("VERSION", version)
    configs = defaultdict(set)
    steps = {}
    events = Counter()
    for row in records:
        events[row["event"]] += 1
        rank = f"{row.get('dp', '?')}/{row.get('tp', '?')}/{row.get('ep', '?')}"
        if row["event"] in ("CONFIG", "CAPACITY"):
            key = json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("pid", "dp", "ep") and not (row["event"] == "CONFIG" and k == "tp")
                },
                sort_keys=True,
            )
            configs[key].add(rank)
        else:
            step = steps.setdefault((row["pid"], row["seq"]), {})
            step[row["event"]] = row
    print("AUDIT", json.dumps(dict(events=events, malformed=malformed, steps=len(steps))))
    print(
        "NATIVE",
        json.dumps(dict(external_backend_entries_by_pid=backend_entries, cpu_binding_failures=binding_failures)),
    )
    for config, ranks in configs.items():
        print("CONFIG", config, "ranks=" + ",".join(sorted(ranks)))
    grouped = defaultdict(list)
    for step in steps.values():
        select = step.get("SELECT", {})
        key = json.dumps(
            {k: v for k, v in select.items() if k not in ("pid", "seq", "dp", "tp", "ep", "event")}, sort_keys=True
        )
        grouped[key].append(step)
    op_sets = {}
    print(
        "FIELDS: sel=selector_tokens p=profile m=metadata d=draft "
        "pre=num_prefills dec=num_decode_tokens s=sample; ranks=dp/tp/ep"
    )
    for key, group in grouped.items():
        ranks = sorted({f"{s['SELECT']['dp']}/{s['SELECT']['tp']}/{s['SELECT']['ep']}" for s in group if "SELECT" in s})
        results = Counter(s.get("END", {}).get("result", "missing_END") for s in group)
        counts = defaultdict(list)
        for step in group:
            for name, n in step.get("OPS", {}).get("counts", {}).items():
                counts[name].append(n)
        op_key = json.dumps({name: [min(ns), max(ns), len(ns)] for name, ns in sorted(counts.items())})
        op_id = op_sets.setdefault(op_key, "O" + str(len(op_sets) + 1)) if counts else "none"
        fields = json.loads(key)
        aliases = dict(
            sel="selector_tokens",
            local="local",
            act="actual",
            maxdp="max_dp",
            pad="pad",
            padded="padded",
            cap="capacity",
            route="route",
            p="profile",
            m="metadata",
            d="draft",
            pre="num_prefills",
            dec="num_decode_tokens",
            skip="skip_compiled",
            s="sample",
        )
        display = " ".join(
            f"{alias}={int(fields[name]) if isinstance(fields.get(name), bool) else fields.get(name)}"
            for alias, name in aliases.items()
        )
        print("STEP", display, "ranks=" + ",".join(ranks), "end=" + json.dumps(results), "ops=" + op_id)
        for error in sorted({s["ERROR"]["message"] for s in group if "ERROR" in s}):
            print("  ERROR", error.replace("\n", " ")[:700])
    for content, op_id in op_sets.items():
        print("OPS", op_id, content)
    if not records:
        raise SystemExit("No MOE_AUDIT records: check imported checkout / MRv1 / startup errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    summarize(parser.parse_args().logs)
