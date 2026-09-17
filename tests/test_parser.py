import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transcript_parser as tp  # noqa: E402


def _entry(kind, **kw):
    e = {"type": kind, "timestamp": "2026-09-17T10:00:00.000Z", "gitBranch": "main",
         "cwd": "/Users/me/proj", "sessionId": "s1"}
    e.update(kw)
    return json.dumps(e)


def _user(text, **kw):
    return _entry("user", message={"role": "user", "content": text}, **kw)


def _assistant(blocks):
    return _entry("assistant", message={"role": "assistant", "content": blocks})


class CleanUserText(unittest.TestCase):
    def test_command_echo_is_noise(self):
        text = ("<command-name>/effort</command-name>\n<command-message>effort</command-message>\n"
                "<command-args></command-args>")
        self.assertEqual(tp.clean_user_text(text), "")

    def test_local_command_output_is_noise(self):
        self.assertEqual(tp.clean_user_text("<local-command-stdout>Set effort to max</local-command-stdout>"), "")

    def test_system_reminder_is_stripped_but_prompt_kept(self):
        text = "<system-reminder>injected\ncontext</system-reminder>Fix the deploy script please"
        self.assertEqual(tp.clean_user_text(text), "Fix the deploy script please")

    def test_sentinel_is_noise(self):
        self.assertEqual(tp.clean_user_text("<<autonomous-loop-dynamic>>"), "")

    def test_real_prompt_untouched(self):
        self.assertEqual(tp.clean_user_text("  Why does pymilvus 2.6.10 break?  "), "Why does pymilvus 2.6.10 break?")


class ParseTranscript(unittest.TestCase):
    def _write(self, lines, trailing_newline=True):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines))
            if trailing_newline:
                f.write("\n")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_basic_turn_with_actions(self):
        path = self._write([
            _user("How do I restart the server?"),
            _assistant([{"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "Run the restart command."},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "./session-rag-server.sh   restart"}},
                        {"type": "tool_use", "name": "Edit", "input": {"file_path": "/Users/me/proj/a.py"}}]),
        ])
        res = tp.parse_transcript(path, "s1")
        self.assertEqual(len(res.turns), 1)
        t = res.turns[0]
        self.assertIn("User: How do I restart the server?", t["text"])
        self.assertIn("Assistant: Run the restart command.", t["text"])
        self.assertIn("- Bash: ./session-rag-server.sh restart", t["text"])
        self.assertIn("- Edit /Users/me/proj/a.py", t["text"])
        self.assertEqual(t["chunk_type"], "turn")
        self.assertEqual(t["turn_index"], 0)
        self.assertEqual(res.cwd, "/Users/me/proj")
        self.assertEqual(res.new_offset, os.path.getsize(path))

    def test_noise_prompts_are_skipped(self):
        path = self._write([
            _user("<command-name>/model</command-name>"),
            _user("<local-command-stdout>Kept model</local-command-stdout>"),
            _user("real question here"),
            _assistant([{"type": "text", "text": "real answer"}]),
        ])
        res = tp.parse_transcript(path, "s1")
        self.assertEqual(len(res.turns), 1)
        self.assertTrue(res.turns[0]["text"].startswith("User: real question here"))

    def test_partial_last_line_is_not_consumed(self):
        lines = [_user("first question"), _assistant([{"type": "text", "text": "first answer"}])]
        path = self._write(lines)
        partial = _user("second question that is still being written")[:40]
        with open(path, "a") as f:
            f.write(partial)  # no newline yet
        res = tp.parse_transcript(path, "s1")
        self.assertEqual(len(res.turns), 1)
        complete_size = os.path.getsize(path) - len(partial)
        self.assertEqual(res.new_offset, complete_size)
        # finish the line and resume from the returned offset
        with open(path, "a") as f:
            f.write(_user("second question that is still being written")[40:] + "\n")
        res2 = tp.parse_transcript(path, "s1", start_offset=res.new_offset)
        self.assertEqual(len(res2.turns), 1)
        self.assertIn("second question", res2.turns[0]["text"])
        self.assertEqual(res2.new_offset, os.path.getsize(path))

    def test_incremental_offsets_and_stable_doc_ids(self):
        path = self._write([_user("q1"), _assistant([{"type": "text", "text": "a1 " * 10}])])
        first = tp.parse_transcript(path, "s1")
        with open(path, "a") as f:
            f.write(_user("q2 second") + "\n" + _assistant([{"type": "text", "text": "a2 " * 10}]) + "\n")
        second = tp.parse_transcript(path, "s1", start_offset=first.new_offset)
        self.assertEqual(len(first.turns), 1)
        self.assertEqual(len(second.turns), 1)
        self.assertNotEqual(first.turns[0]["doc_id"], second.turns[0]["doc_id"])
        self.assertLess(first.turns[0]["turn_index"], second.turns[0]["turn_index"])
        again = tp.parse_transcript(path, "s1")
        self.assertEqual([t["doc_id"] for t in again.turns],
                         [first.turns[0]["doc_id"], second.turns[0]["doc_id"]])

    def test_long_turn_is_chunked_not_truncated(self):
        answer = "\n\n".join(f"Paragraph {i} " + ("lorem ipsum " * 40) for i in range(60))
        path = self._write([_user("explain everything"), _assistant([{"type": "text", "text": answer}])])
        res = tp.parse_transcript(path, "s1")
        self.assertGreater(len(res.turns), 1)
        for i, t in enumerate(res.turns):
            self.assertLessEqual(len(t["text"]), tp.MAX_CHUNK_CHARS)
            self.assertTrue(t["text"].startswith("User: explain everything"))
            self.assertIn(f"(part {i + 1}/{len(res.turns)})", t["text"])
        self.assertIn("Paragraph 59", res.turns[-1]["text"])
        self.assertEqual(len({t["doc_id"] for t in res.turns}), len(res.turns))

    def test_summary_and_ai_title(self):
        path = self._write([
            json.dumps({"type": "summary", "summary": "Fixing sorting in icm-client", "leafUuid": "x"}),
            _user("hello there friend"),
            _assistant([{"type": "text", "text": "hi"}]),
            json.dumps({"type": "ai-title", "aiTitle": "Sorting fix session", "sessionId": "s1"}),
        ])
        res = tp.parse_transcript(path, "s1")
        types = [t["chunk_type"] for t in res.turns]
        self.assertEqual(types.count("summary"), 2)
        self.assertIn("Session Title: Sorting fix session", [t["text"] for t in res.turns])

    def test_subagent_chunk_type(self):
        path = self._write([_user("Explore the repo thoroughly"), _assistant([{"type": "text", "text": "Report..."}])])
        res = tp.parse_transcript(path, "parent-session", chunk_type="subagent")
        self.assertEqual(res.turns[0]["chunk_type"], "subagent")
        self.assertEqual(res.turns[0]["session_id"], "parent-session")

    def test_offset_beyond_size_restarts(self):
        path = self._write([_user("hello world question"), _assistant([{"type": "text", "text": "answer"}])])
        res = tp.parse_transcript(path, "s1", start_offset=10 ** 9)
        self.assertEqual(len(res.turns), 1)


if __name__ == "__main__":
    unittest.main()
