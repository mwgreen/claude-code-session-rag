"""Bedrock backend: model resolution, batching, retry and the index identity
check, against a stubbed bedrock-runtime client. Nothing touches ~/.session-rag.

Opt-in integration test against the real service: RUN_BEDROCK_IT=1 (model from
SESSION_RAG_BEDROCK_EMBED_MODEL, default us.cohere.embed-v4:0).
"""

import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from botocore.exceptions import ClientError
    HAVE_BOTO = True
except ImportError:
    HAVE_BOTO = False

import bedrock_backend as bb  # noqa: E402
import embedder  # noqa: E402

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def fake_vector(text: str, dim: int):
    vec = [0.0] * dim
    for tok in _TOKEN.findall(text.lower()):
        vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class StubBedrock:
    def __init__(self, failures=None):
        self.invocations = []
        self.failures = list(failures or [])

    def invoke_model(self, modelId, body, **kwargs):
        req = json.loads(body)
        self.invocations.append((modelId, req))
        if self.failures:
            raise self.failures.pop(0)
        if "texts" in req:
            payload = {"embeddings": {"float": [fake_vector(t, req["output_dimension"]) for t in req["texts"]]}}
        else:
            payload = {"embedding": fake_vector(req["inputText"], req["dimensions"]), "inputTextTokenCount": 3}
        return {"body": io.BytesIO(json.dumps(payload).encode()), "ResponseMetadata": {"HTTPHeaders": {}}}


class EnvCase(unittest.TestCase):
    VARS = ("SESSION_RAG_BACKEND", "SESSION_RAG_BEDROCK_EMBED_MODEL", "SESSION_RAG_BEDROCK_EMBED_DIM",
            "SESSION_RAG_MODEL")

    def setUp(self):
        self.saved_env = {k: os.environ.get(k) for k in self.VARS}
        for k in self.VARS:
            os.environ.pop(k, None)
        self.saved_bb = (bb._sleep, bb._random)
        self.sleeps = []
        bb._sleep, bb._random = self.sleeps.append, (lambda: 1.0)
        bb.USAGE.reset()

    def tearDown(self):
        for k, v in self.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        bb._sleep, bb._random = self.saved_bb


class ResolutionTests(EnvCase):
    def test_default_backend_is_mlx_and_unchanged(self):
        self.assertEqual(embedder.resolve_backend(), "mlx")
        spec = embedder.get_spec()
        self.assertEqual(spec.name, embedder.resolve_model_name())
        self.assertIn(spec.name, embedder.REGISTRY)
        self.assertIsInstance(embedder.make_embedder(spec), embedder.Embedder)

    def test_bedrock_default_is_cohere_v4_1024(self):
        os.environ["SESSION_RAG_BACKEND"] = "bedrock"
        spec = embedder.get_spec()
        self.assertEqual((spec.name, spec.model_id, spec.dim), ("bedrock-cohere-v4", "us.cohere.embed-v4:0", 1024))
        self.assertEqual(embedder.backend_of(spec), "bedrock")
        self.assertTrue(spec.is_downloaded())

    def test_bedrock_model_and_dim_from_env(self):
        os.environ.update(SESSION_RAG_BACKEND="bedrock", SESSION_RAG_BEDROCK_EMBED_MODEL="amazon.titan-embed-text-v2:0",
                          SESSION_RAG_BEDROCK_EMBED_DIM="512")
        spec = embedder.get_spec()
        self.assertEqual((spec.name, spec.dim), ("bedrock-titan-v2", 512))

    def test_bare_cohere_id_rejected_with_hint(self):
        os.environ.update(SESSION_RAG_BACKEND="bedrock", SESSION_RAG_BEDROCK_EMBED_MODEL="cohere.embed-v4:0")
        with self.assertRaisesRegex(ValueError, "us.cohere.embed-v4:0"):
            embedder.get_spec()

    def test_unknown_backend_rejected(self):
        os.environ["SESSION_RAG_BACKEND"] = "cuda"
        with self.assertRaises(ValueError):
            embedder.get_spec()


@unittest.skipUnless(HAVE_BOTO, "boto3 not installed")
class AdapterTests(EnvCase):
    def make(self, model="bedrock-cohere-v4", failures=None):
        os.environ["SESSION_RAG_BACKEND"] = "bedrock"
        self.stub = StubBedrock(failures)
        return embedder.BedrockEmbedderAdapter(embedder.get_spec(model), client=self.stub)

    def test_documents_batched_by_96_queries_use_search_query(self):
        emb = self.make()
        vecs = emb.embed_documents([f"turn {i}" for i in range(200)])
        self.assertEqual(len(vecs), 200)
        self.assertEqual([len(r["texts"]) for _, r in self.stub.invocations], [96, 96, 8])
        self.assertEqual({r["input_type"] for _, r in self.stub.invocations}, {"search_document"})
        emb.embed_query("what did we decide")
        self.assertEqual(self.stub.invocations[-1][1]["input_type"], "search_query")
        self.assertEqual(emb.identity(), {"backend": "bedrock", "model_name": "bedrock-cohere-v4",
                                          "model_id": "us.cohere.embed-v4:0", "embed_dim": 1024})

    def test_titan_one_text_per_call(self):
        emb = self.make("bedrock-titan-v2")
        emb.embed_documents(["a", "b", "c"])
        self.assertEqual(len(self.stub.invocations), 3)
        self.assertTrue(all(r["normalize"] is True for _, r in self.stub.invocations))

    def test_throttling_is_retried(self):
        err = ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "InvokeModel")
        emb = self.make(failures=[err, err])
        self.assertEqual(len(emb.embed_query("x")), 1024)
        self.assertEqual(len(self.stub.invocations), 3)
        self.assertEqual(self.sleeps, [0.5, 1.0])


@unittest.skipUnless(HAVE_BOTO, "boto3 not installed")
class IdentityTests(EnvCase):
    def setUp(self):
        super().setUp()
        import rag_engine
        self.re = rag_engine
        self.saved = (rag_engine._spec, rag_engine._embedder, rag_engine.IDENTITY_FILE)
        self.tmp = Path(tempfile.mkdtemp(prefix="session-rag-bedrock-"))
        self.db = str(self.tmp / "milvus.db")
        rag_engine.IDENTITY_FILE = self.tmp / "model_identity.json"
        os.environ["SESSION_RAG_BACKEND"] = "bedrock"

    def tearDown(self):
        self.re._spec, self.re._embedder, self.re.IDENTITY_FILE = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def use(self, name):
        spec = self.re.use_model(name)
        self.stub = StubBedrock()
        self.re._embedder = embedder.BedrockEmbedderAdapter(spec, client=self.stub)

    def turns(self, n=3):
        out = []
        for i in range(n):
            out.append({"doc_id": f"s1:{i}", "text": f"User: how does WarehouseRefreshOrchestrator work {i}",
                        "session_id": "s1", "transcript_file": "/t/s1.jsonl", "turn_index": i,
                        "timestamp": "2026-10-14T00:00:00Z", "git_branch": "main", "chunk_type": "turn",
                        "project_root": "/home/x/repo"})
        return out

    def test_index_search_then_model_switch_refused(self):
        self.use("bedrock-cohere-v4")
        self.re._check_model_identity(self.db)
        self.assertEqual(self.re.add_turns(self.turns(), db_path=self.db), 3)
        hits = self.re.search("WarehouseRefreshOrchestrator", n=2, db_path=self.db, project_root="/home/x/repo")
        self.assertTrue(hits)
        self.assertEqual(self.stub.invocations[-1][1]["input_type"], "search_query")
        self.assertEqual(json.loads(self.re.IDENTITY_FILE.read_text())["backend"], "bedrock")

        self.use("bedrock-titan-v2")  # same 1024 dims, different model
        with self.assertRaisesRegex(RuntimeError, "Model mismatch.*migrate-model --to bedrock-titan-v2"):
            self.re._check_model_identity(self.db)

    def test_mlx_built_index_refused_by_bedrock(self):
        # Identity as the Mac writes it (no "backend" key), with vectors in the index.
        self.use("bedrock-cohere-v4")
        self.re.add_turns(self.turns(), db_path=self.db)
        self.re.IDENTITY_FILE.write_text(json.dumps(
            {"model_name": "qwen3", "model_id": "mlx-community/Qwen3-Embedding-0.6B-8bit", "embed_dim": 1024}))
        with self.assertRaisesRegex(RuntimeError, "built with mlx 'qwen3'"):
            self.re._check_model_identity(self.db)

    def test_other_dimension_refused(self):
        self.use("bedrock-cohere-v4")
        self.re.add_turns(self.turns(), db_path=self.db)
        self.re.stamp_identity()
        os.environ["SESSION_RAG_BEDROCK_EMBED_DIM"] = "512"
        self.use("bedrock-cohere-v4")
        with self.assertRaisesRegex(RuntimeError, "1024d.*512d"):
            self.re._check_model_identity(self.db)

    def test_empty_index_is_restamped(self):
        self.re.IDENTITY_FILE.write_text(json.dumps({"model_name": "embeddinggemma", "embed_dim": 768}))
        self.use("bedrock-titan-v2")
        self.re._check_model_identity(self.db)
        self.assertEqual(json.loads(self.re.IDENTITY_FILE.read_text())["model_name"], "bedrock-titan-v2")


class NoMlxImportTests(unittest.TestCase):
    def test_bedrock_backend_never_imports_mlx(self):
        if not HAVE_BOTO:
            self.skipTest("boto3 not installed")
        home = tempfile.mkdtemp(prefix="session-rag-home-")
        try:
            code = ("import sys, http_server, rag_engine\n"
                    "rag_engine.load_model()\n"
                    "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('mlx', 'mlx_embeddings', 'mlx_lm'))\n"
                    "print(rag_engine.get_backend(), rag_engine.get_model_name(), bad)\n"
                    "sys.exit(1 if bad else 0)\n")
            env = dict(os.environ, SESSION_RAG_BACKEND="bedrock", HOME=home, PYTHONPATH=str(ROOT))
            proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr[-2000:])
            self.assertIn("bedrock bedrock-cohere-v4 []", proc.stdout)
        finally:
            shutil.rmtree(home, ignore_errors=True)


@unittest.skipUnless(os.getenv("RUN_BEDROCK_IT") == "1", "set RUN_BEDROCK_IT=1 to call Bedrock")
class BedrockIntegrationTest(unittest.TestCase):
    def test_real_embeddings(self):
        saved = os.environ.get("SESSION_RAG_BACKEND")
        os.environ["SESSION_RAG_BACKEND"] = "bedrock"
        try:
            emb = embedder.make_embedder(embedder.get_spec())
            docs = emb.embed_documents(["We fixed the JWT user resolution by reading the subject claim.",
                                        "Lunch options near the office"])
            query = emb.embed_query("how was JWT user resolution fixed")
        finally:
            if saved is None:
                os.environ.pop("SESSION_RAG_BACKEND", None)
            else:
                os.environ["SESSION_RAG_BACKEND"] = saved

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
        self.assertEqual(len(query), emb.spec.dim)
        self.assertGreater(cos(query, docs[0]), cos(query, docs[1]))


if __name__ == "__main__":
    unittest.main()
