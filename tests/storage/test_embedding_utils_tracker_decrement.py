# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import types

import pytest

from openviking.storage.queuefs.embedding_tracker import EmbeddingTaskTracker
from openviking.utils import embedding_utils
from tests.utils.mock_context import make_test_ctx


class DummyQueue:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.enqueued: list[object] = []

    async def enqueue(self, msg: object) -> str:
        if self.fail:
            raise RuntimeError("enqueue failed")
        self.enqueued.append(msg)
        return "dummy"


class DummyQueueManager:
    EMBEDDING = "Embedding"

    def __init__(self, queue: DummyQueue):
        self._queue = queue

    def get_queue(self, _name: str) -> DummyQueue:
        return self._queue


def _fake_from_context(ctx) -> object | None:
    text = ctx.get_vectorization_text()
    if not text:
        return None
    return types.SimpleNamespace(semantic_msg_id=None)


@pytest.mark.asyncio
async def test_vectorize_directory_meta_decrement_when_no_msgs(monkeypatch):
    queue = DummyQueue()
    monkeypatch.setattr(embedding_utils, "get_queue_manager", lambda: DummyQueueManager(queue))
    monkeypatch.setattr(embedding_utils.EmbeddingMsgConverter, "from_context", _fake_from_context)

    semantic_msg_id = "test_dir_no_msgs"
    tracker = EmbeddingTaskTracker.get_instance()
    await tracker.remove(semantic_msg_id)

    done = asyncio.Event()

    async def on_complete():
        done.set()

    await tracker.register(semantic_msg_id=semantic_msg_id, total_count=2, on_complete=on_complete)

    await embedding_utils.vectorize_directory_meta(
        uri="viking://resources/dir",
        abstract="",
        overview="",
        ctx=make_test_ctx(),
        semantic_msg_id=semantic_msg_id,
    )

    assert done.is_set()
    assert queue.enqueued == []
    assert await tracker.get_status(semantic_msg_id) is None


@pytest.mark.asyncio
async def test_vectorize_directory_meta_decrement_only_missing(monkeypatch):
    queue = DummyQueue()
    monkeypatch.setattr(embedding_utils, "get_queue_manager", lambda: DummyQueueManager(queue))
    monkeypatch.setattr(embedding_utils.EmbeddingMsgConverter, "from_context", _fake_from_context)

    semantic_msg_id = "test_dir_one_msg"
    tracker = EmbeddingTaskTracker.get_instance()
    await tracker.remove(semantic_msg_id)

    done = asyncio.Event()

    async def on_complete():
        done.set()

    await tracker.register(semantic_msg_id=semantic_msg_id, total_count=2, on_complete=on_complete)

    await embedding_utils.vectorize_directory_meta(
        uri="viking://resources/dir",
        abstract="",
        overview="hello",
        ctx=make_test_ctx(),
        semantic_msg_id=semantic_msg_id,
    )

    status = await tracker.get_status(semantic_msg_id)
    assert not done.is_set()
    assert status is not None and status["remaining"] == 1
    assert len(queue.enqueued) == 1

    await tracker.decrement(semantic_msg_id)
    assert done.is_set()
    assert await tracker.get_status(semantic_msg_id) is None


@pytest.mark.asyncio
async def test_vectorize_file_decrement_when_skipped(monkeypatch):
    queue = DummyQueue()
    monkeypatch.setattr(embedding_utils, "get_queue_manager", lambda: DummyQueueManager(queue))
    monkeypatch.setattr(embedding_utils.EmbeddingMsgConverter, "from_context", _fake_from_context)
    monkeypatch.setattr(embedding_utils, "get_viking_fs", lambda: object())

    semantic_msg_id = "test_file_skipped"
    tracker = EmbeddingTaskTracker.get_instance()
    await tracker.remove(semantic_msg_id)

    done = asyncio.Event()

    async def on_complete():
        done.set()

    await tracker.register(semantic_msg_id=semantic_msg_id, total_count=1, on_complete=on_complete)

    await embedding_utils.vectorize_file(
        file_path="viking://resources/file.bin",
        summary_dict={"name": "file.bin"},
        parent_uri="viking://resources",
        ctx=make_test_ctx(),
        semantic_msg_id=semantic_msg_id,
    )

    assert done.is_set()
    assert queue.enqueued == []
    assert await tracker.get_status(semantic_msg_id) is None
