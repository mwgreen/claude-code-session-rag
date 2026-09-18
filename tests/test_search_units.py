import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import plan_batches  # noqa: E402
from fts_hybrid import FTSIndex, build_match_query, rrf_merge  # noqa: E402
from index_state import IndexState, SlugMap, atomic_write_json  # noqa: E402
import indexer  # noqa: E402


class MatchQuery(unittest.TestCase):
    def test_punctuation_is_safe(self):
        q = build_match_query("error in deploy script: can't find module")
        self.assertEqual(q, '"error"* OR "deploy"* OR "script"* OR "find"* OR "module"*')

    def test_identifiers_kept_whole(self):
        self.assertEqual(build_match_query("add_turns_async"), '"add_turns_async"*')

    def test_only_stopwords(self):
        self.assertIsNone(build_match_query("the a to"))


class FTSRoundTrip(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = os.path.join(self.dir, "milvus.db")
        self.fts = FTSIndex("t", ["session_id", "project_root", "turn_index"])

    def test_insert_search_prefix_filter(self):
        conn = self.fts.connection(self.db)
        n = self.fts.insert(conn, [
            {"doc_id": "a", "content": "User: fix add_turns_async race\n\nAssistant: use a lock",
             "session_id": "s1", "project_root": "/repo", "turn_index": 1},
            {"doc_id": "b", "content": "unrelated cooking recipe", "session_id": "s2",
             "project_root": "/repo/sub", "turn_index": 2},
            {"doc_id": "a", "content": "duplicate", "session_id": "s1", "project_root": "/repo", "turn_index": 3},
        ])
        self.assertEqual(n, 2)
        conn.close()
        hits = self.fts.search("add_turns_async lock", db_path=self.db)
        self.assertEqual([h["doc_id"] for h in hits], ["a"])
        hits = self.fts.search("recipe", prefix_filters={"project_root": "/repo"}, db_path=self.db)
        self.assertEqual([h["doc_id"] for h in hits], ["b"])
        hits = self.fts.search("recipe", filters={"project_root": "/repo"}, db_path=self.db)
        self.assertEqual(hits, [])

    def test_schema_change_rebuilds(self):
        conn = self.fts.connection(self.db)
        self.fts.insert(conn, [{"doc_id": "a", "content": "x y z", "session_id": "s", "project_root": "", "turn_index": 0}])
        conn.close()
        other = FTSIndex("t", ["session_id", "project_root", "turn_index"], tokenizer="unicode61")
        conn = other.connection(self.db)
        self.assertEqual(other.count(conn), 0)
        conn.close()


class RRF(unittest.TestCase):
    def test_merge_keeps_vector_fields(self):
        vec = [{"doc_id": "a", "similarity": 0.9}, {"doc_id": "b", "similarity": 0.8}]
        fts = [{"doc_id": "b", "bm25": -3.0}, {"doc_id": "c", "bm25": -2.0}]
        merged = rrf_merge(vec, fts, n=10)
        self.assertEqual(merged[0]["doc_id"], "b")
        self.assertEqual(merged[0]["similarity"], 0.8)
        self.assertIn("_rrf_score", merged[0])
        self.assertEqual({m["doc_id"] for m in merged}, {"a", "b", "c"})


class Batching(unittest.TestCase):
    def test_budget_respected(self):
        texts = ["x" * n for n in (10, 50, 9000, 24000, 300, 3000, 6000, 100)]
        batches = plan_batches(texts, max_tokens=2048, token_budget=4096, max_batch=4)
        covered = sorted(i for b in batches for i in b)
        self.assertEqual(covered, list(range(len(texts))))
        for b in batches:
            longest = max(min(max(8, len(texts[i]) // 3), 2048) for i in b)
            self.assertLessEqual(len(b) * longest, 4096)


class State(unittest.TestCase):
    def test_atomic_write_and_reload(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "index_state.json")
        st = IndexState(p)
        st.set_offset("/a.jsonl", 10, "/repo")
        self.assertTrue(st.save())
        self.assertFalse(st.save())
        again = IndexState(p)
        self.assertEqual(again.get_offset("/a.jsonl"), 10)
        self.assertEqual(again.get_project_root("/a.jsonl"), "/repo")
        self.assertEqual(os.listdir(d), ["index_state.json"])  # no temp files left behind

    def test_corrupt_file_starts_empty(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "index_state.json")
        with open(p, "w") as f:
            f.write('{"transcripts": {"/a.jsonl": {"last_byte_off')
        self.assertEqual(IndexState(p).get_offset("/a.jsonl"), 0)

    def test_slug_map(self):
        d = tempfile.mkdtemp()
        sm = SlugMap(os.path.join(d, "slug_map.json"))
        self.assertTrue(sm.register("/Users/me/git-repos/proj.x"))
        self.assertFalse(sm.register("/Users/me/git-repos/proj.x"))
        self.assertEqual(sm.get("-Users-me-git-repos-proj-x"), "/Users/me/git-repos/proj.x")


class Classify(unittest.TestCase):
    def test_layouts(self):
        base = str(indexer.CLAUDE_PROJECTS)
        self.assertEqual(indexer.classify_transcript(f"{base}/-Users-me-proj/abc.jsonl"),
                         ("-Users-me-proj", "abc", "turn"))
        self.assertEqual(indexer.classify_transcript(f"{base}/-Users-me-proj/abc/subagents/agent-1.jsonl"),
                         ("-Users-me-proj", "abc", "subagent"))
        self.assertEqual(indexer.classify_transcript(f"{base}/-Users-me-proj/archived-sessions/old.jsonl"),
                         ("-Users-me-proj", "old", "turn"))
        self.assertEqual(indexer.classify_transcript("/tmp/elsewhere/xyz.jsonl"), (None, "xyz", "turn"))

    def test_enqueue_normalises_symlinked_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_dir = os.path.join(tmp, "real"); os.mkdir(real_dir)
            link_dir = os.path.join(tmp, "link"); os.symlink(real_dir, link_dir)
            real = os.path.join(real_dir, "s.jsonl"); open(real, "w").close()
            idx = indexer.Indexer(os.path.join(tmp, "milvus.db"))
            idx.enqueue(os.path.join(link_dir, "s.jsonl"))
            idx.enqueue(real)
            self.assertEqual(list(idx._live), [os.path.realpath(real)])


if __name__ == "__main__":
    unittest.main()
