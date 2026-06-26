"""Tests for paper_reader.py."""

from paper_reader import (
    TapeReader,
    _summarize_tool_input,
    _parse_jsonl_entry,
)


class TestSummarizeToolInput:
    def test_read(self):
        assert _summarize_tool_input("Read", {"file_path": "/foo.py"}) == "/foo.py"

    def test_write(self):
        assert _summarize_tool_input("Write", {"file_path": "/bar.py"}) == "/bar.py"

    def test_edit(self):
        assert _summarize_tool_input("Edit", {"file_path": "/baz.py"}) == "/baz.py"

    def test_bash(self):
        assert _summarize_tool_input("Bash", {"command": "ls -la"}) == "ls -la"

    def test_bash_truncated(self):
        result = _summarize_tool_input("Bash", {"command": "x" * 300})
        assert len(result) == 200

    def test_grep(self):
        assert _summarize_tool_input("Grep", {"pattern": "TODO"}) == "pattern=TODO"

    def test_glob(self):
        assert _summarize_tool_input("Glob", {"pattern": "**/*.py"}) == "pattern=**/*.py"

    def test_agent(self):
        assert _summarize_tool_input("Agent", {"description": "find bugs"}) == "find bugs"

    def test_fallback_known_key(self):
        assert "query=test" in _summarize_tool_input("Unknown", {"query": "test"})

    def test_fallback_no_known_key(self):
        result = _summarize_tool_input("Unknown", {"foo": "bar"})
        assert "foo" in result

    def test_non_dict_input(self):
        result = _summarize_tool_input("Read", "just a string")
        assert result == "just a string"


class TestParseJsonlEntry:
    def _user(self, content):
        return {
            "type": "user",
            "timestamp": "2026-03-09T10:00:00Z",
            "message": {"role": "user", "content": content},
        }

    def _assistant(self, content, usage=None):
        return {
            "type": "assistant",
            "timestamp": "2026-03-09T10:01:00Z",
            "message": {
                "role": "assistant",
                "content": content,
                "usage": usage or {},
            },
        }

    def test_non_dict_returns_none(self):
        assert _parse_jsonl_entry(None) is None
        assert _parse_jsonl_entry([]) is None
        assert _parse_jsonl_entry("string") is None

    def test_unknown_type_returns_none(self):
        assert _parse_jsonl_entry({"type": "system"}) is None

    def test_user_text_string(self):
        entry = _parse_jsonl_entry(self._user("hello world"))
        assert entry is not None
        assert entry.type == "user"
        assert entry.text_content == "hello world"

    def test_user_text_block(self):
        entry = _parse_jsonl_entry(self._user([{"type": "text", "text": "hello"}]))
        assert entry.text_content == "hello"

    def test_user_tool_result(self):
        content = [{"type": "tool_result", "tool_use_id": "tu-1", "content": "output", "is_error": False}]
        entry = _parse_jsonl_entry(self._user(content))
        assert len(entry.tool_results) == 1
        assert entry.tool_results[0].tool_use_id == "tu-1"
        assert not entry.tool_results[0].is_error

    def test_user_tool_result_error(self):
        content = [{"type": "tool_result", "tool_use_id": "tu-2", "content": "err", "is_error": True}]
        entry = _parse_jsonl_entry(self._user(content))
        assert entry.tool_results[0].is_error

    def test_assistant_text(self):
        entry = _parse_jsonl_entry(self._assistant([{"type": "text", "text": "ok"}]))
        assert entry is not None
        assert entry.type == "assistant"
        assert entry.text_content == "ok"

    def test_assistant_tool_use(self):
        content = [{"type": "tool_use", "id": "tu-1", "name": "Read", "input": {"file_path": "/x.py"}}]
        entry = _parse_jsonl_entry(self._assistant(content))
        assert len(entry.tool_uses) == 1
        assert entry.tool_uses[0].name == "Read"
        assert entry.tool_uses[0].input_summary == "/x.py"

    def test_assistant_token_usage(self):
        usage = {
            "input_tokens": 100, "output_tokens": 50,
            "cache_read_input_tokens": 80, "cache_creation_input_tokens": 10,
        }
        entry = _parse_jsonl_entry(self._assistant([], usage))
        assert entry.token_usage.input_tokens == 100
        assert entry.token_usage.output_tokens == 50
        assert entry.token_usage.cache_read == 80
        assert entry.token_usage.cache_creation == 10


class TestTapeReaderFilesystemFallback:
    def test_falls_back_to_jsonl_dir_when_no_paperd(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        jsonl_dir = tmp_path / "projects" / "-test"
        jsonl_dir.mkdir(parents=True)
        (jsonl_dir / "session-aaa.jsonl").write_text("")
        (jsonl_dir / "session-bbb.jsonl").write_text("")

        reader = TapeReader()
        reader._jsonl_dir = jsonl_dir
        reader._base_url = None

        sessions = reader.list_sessions()
        assert set(sessions) == {"session-aaa", "session-bbb"}

    def test_returns_empty_when_no_paperd_and_no_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        reader = TapeReader()
        reader._jsonl_dir = tmp_path / "nonexistent"
        reader._base_url = None
        assert reader.list_sessions() == []

    def test_jsonl_files_ordered_by_mtime(self, tmp_path, monkeypatch):
        import time
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        jsonl_dir = tmp_path / "projects" / "-test"
        jsonl_dir.mkdir(parents=True)
        a = jsonl_dir / "session-a.jsonl"
        b = jsonl_dir / "session-b.jsonl"
        a.write_text("")
        time.sleep(0.01)
        b.write_text("")

        reader = TapeReader()
        reader._jsonl_dir = jsonl_dir
        reader._base_url = None

        sessions = reader.list_sessions()
        assert sessions == ["session-a", "session-b"]
