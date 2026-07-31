"""Tests for Live Verification Streaming API."""

import pytest

from codeverify_core.streaming_verification import (
    IncrementalDiff,
    SessionStatus,
    StreamEvent,
    StreamEventType,
    StreamingSessionConfig,
    StreamingSessionPool,
    StreamingVerificationSession,
    VerificationStage,
    get_streaming_pool,
    reset_streaming_pool,
)


class TestIncrementalDiff:
    """Tests for IncrementalDiff."""

    def test_code_hash(self):
        diff = IncrementalDiff(file_path="a.py", old_code="", new_code="x = 1")
        assert len(diff.code_hash) == 16

    def test_is_meaningful_detects_real_change(self):
        diff = IncrementalDiff(file_path="a.py", old_code="x = 1", new_code="x = 2")
        assert diff.is_meaningful is True

    def test_is_meaningful_ignores_whitespace(self):
        diff = IncrementalDiff(file_path="a.py", old_code="x = 1", new_code="  x = 1  ")
        assert diff.is_meaningful is False


class TestStreamEvent:
    def test_to_sse(self):
        evt = StreamEvent(
            event_type=StreamEventType.HEARTBEAT,
            data={"msg": "ping"},
            session_id="s1",
            sequence=1,
        )
        sse = evt.to_sse()
        assert sse.startswith("data: ")
        assert '"type": "heartbeat"' in sse

    def test_to_dict(self):
        evt = StreamEvent(
            event_type=StreamEventType.FINDING,
            data={"line": 5},
            session_id="s2",
            sequence=2,
        )
        d = evt.to_dict()
        assert d["event_type"] == "finding"
        assert d["session_id"] == "s2"


class TestStreamingVerificationSession:
    def test_session_defaults(self):
        session = StreamingVerificationSession()
        assert session.status == SessionStatus.IDLE
        assert len(session.session_id) > 0

    def test_custom_session_id(self):
        session = StreamingVerificationSession(session_id="my-session")
        assert session.session_id == "my-session"

    @pytest.mark.asyncio
    async def test_verify_incremental_returns_events(self):
        session = StreamingVerificationSession()
        diff = IncrementalDiff(
            file_path="test.py",
            old_code="",
            new_code="x = eval('code')\n",
        )
        events = await session.verify_incremental(diff)
        assert len(events) > 0
        assert any(e.event_type == StreamEventType.COMPLETE for e in events)

    @pytest.mark.asyncio
    async def test_verify_skips_whitespace_only(self):
        session = StreamingVerificationSession()
        diff = IncrementalDiff(
            file_path="test.py",
            old_code="x = 1",
            new_code="  x = 1  ",
        )
        events = await session.verify_incremental(diff)
        complete = [e for e in events if e.event_type == StreamEventType.COMPLETE]
        assert len(complete) == 1
        assert complete[0].data.get("skipped") is True

    @pytest.mark.asyncio
    async def test_verify_finds_pattern_issues(self):
        session = StreamingVerificationSession()
        diff = IncrementalDiff(
            file_path="test.py",
            old_code="",
            new_code="result = eval(input())\n",
        )
        events = await session.verify_incremental(diff)
        findings = [e for e in events if e.event_type == StreamEventType.FINDING]
        assert len(findings) >= 1
        assert any("eval" in f.data.get("message", "") for f in findings)

    @pytest.mark.asyncio
    async def test_stage_events_emitted(self):
        config = StreamingSessionConfig(stages=[VerificationStage.PATTERN])
        session = StreamingVerificationSession(config=config)
        diff = IncrementalDiff(file_path="t.py", old_code="", new_code="x = 1\n")
        events = await session.verify_incremental(diff)
        types = [e.event_type for e in events]
        assert StreamEventType.STAGE_START in types
        assert StreamEventType.STAGE_COMPLETE in types

    @pytest.mark.asyncio
    async def test_on_event_callback(self):
        collected: list[StreamEvent] = []
        session = StreamingVerificationSession()
        diff = IncrementalDiff(file_path="t.py", old_code="", new_code="x = 1\n")
        await session.verify_incremental(diff, on_event=collected.append)
        assert len(collected) > 0

    def test_z3_scope_management(self):
        session = StreamingVerificationSession()
        session.push_z3_scope()
        session.push_z3_scope()
        assert session._z3_scope_depth == 2
        session.pop_z3_scope()
        assert session._z3_scope_depth == 1
        session.reset_z3_scope()
        assert session._z3_scope_depth == 0

    def test_close_session(self):
        session = StreamingVerificationSession()
        session.push_z3_scope()
        session.close()
        assert session.status == SessionStatus.CLOSED
        assert session._z3_scope_depth == 0

    def test_sequence_increments(self):
        session = StreamingVerificationSession()
        assert session._next_seq() == 1
        assert session._next_seq() == 2


class TestStreamingSessionPool:
    def test_create_session(self):
        pool = StreamingSessionPool(max_sessions=5)
        session = pool.create_session()
        assert pool.active_count == 1
        assert pool.get_session(session.session_id) is session

    def test_close_session(self):
        pool = StreamingSessionPool()
        session = pool.create_session()
        assert pool.close_session(session.session_id) is True
        assert pool.get_session(session.session_id) is None

    def test_close_nonexistent(self):
        pool = StreamingSessionPool()
        assert pool.close_session("nope") is False

    def test_max_sessions_evicts_oldest(self):
        pool = StreamingSessionPool(max_sessions=2)
        s1 = pool.create_session()
        pool.create_session()
        pool.create_session()
        assert pool.active_count == 2
        assert pool.get_session(s1.session_id) is None

    def test_get_stats(self):
        pool = StreamingSessionPool()
        pool.create_session()
        stats = pool.get_stats()
        assert stats["active_sessions"] == 1
        assert len(stats["sessions"]) == 1

    def test_singleton_pool(self):
        reset_streaming_pool()
        p1 = get_streaming_pool()
        p2 = get_streaming_pool()
        assert p1 is p2
        reset_streaming_pool()


class TestStreamingEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_code(self):
        session = StreamingVerificationSession()
        diff = IncrementalDiff(file_path="t.py", old_code="", new_code="")
        events = await session.verify_incremental(diff)
        assert any(e.data.get("skipped") for e in events)

    @pytest.mark.asyncio
    async def test_go_language_patterns(self):
        config = StreamingSessionConfig(
            language="go",
            stages=[VerificationStage.PATTERN],
        )
        session = StreamingVerificationSession(config=config)
        diff = IncrementalDiff(
            file_path="main.go",
            old_code="",
            new_code="_ = doSomething()\npanic(err)\n",
        )
        events = await session.verify_incremental(diff)
        findings = [e for e in events if e.event_type == StreamEventType.FINDING]
        assert len(findings) >= 1

    @pytest.mark.asyncio
    async def test_java_language_patterns(self):
        config = StreamingSessionConfig(
            language="java",
            stages=[VerificationStage.PATTERN],
        )
        session = StreamingVerificationSession(config=config)
        diff = IncrementalDiff(
            file_path="Main.java",
            old_code="",
            new_code="System.out.println(x);\n",
        )
        events = await session.verify_incremental(diff)
        findings = [e for e in events if e.event_type == StreamEventType.FINDING]
        assert len(findings) >= 1
