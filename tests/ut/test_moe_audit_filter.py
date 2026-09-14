"""Standalone stdlib-only test of compact log parsing."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "tools/moe_audit/filter.py"
spec = importlib.util.spec_from_file_location("moe_audit_filter", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FilterTest(unittest.TestCase):
    def test_merge_all_ranks_and_incomplete_records(self):
        rows = []
        for rank in (0, 1):
            base = dict(pid=100 + rank, seq=1, dp=0, tp=rank, ep=rank)
            rows.append(dict(base, event="SELECT", route="ALLTOALL", selector_tokens=256, sample=1))
            if rank == 0:
                rows.append(dict(base, event="END", result="returned"))
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "prefill.log"
            log.write_text(
                "\n".join("(Worker) [MOE_AUDIT] " + json.dumps(row) for row in rows)
                + '\n[MOE_AUDIT] {broken\n[MOE_AUDIT] {"not_event":1}\n'
                + "(Worker pid=123) Enter external FX backend graph_id=1 pid=123; saved\n"
            )
            with contextlib.redirect_stdout(output):
                module.summarize([log])
        text = output.getvalue()
        self.assertIn('"malformed": 2', text)
        self.assertIn("ranks=0/0/0,0/1/1", text)
        self.assertIn("missing_END", text)
        self.assertIn("route=ALLTOALL", text)
        self.assertIn('"external_backend_entries_by_pid": {"123": 1}', text)

    def test_no_records_fails_visibly(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "prefill.log"
            log.write_text("nothing\n")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                module.summarize([log])


if __name__ == "__main__":
    unittest.main()
