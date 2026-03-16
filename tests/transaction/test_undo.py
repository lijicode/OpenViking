# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for undo log and rollback executor."""

import uuid

from openviking.storage.transaction.undo import UndoEntry, execute_rollback

from .conftest import VECTOR_DIM, _mkdir_ok, file_exists


class TestUndoEntry:
    def test_to_dict(self):
        entry = UndoEntry(sequence=0, op_type="fs_mv", params={"src": "/a", "dst": "/b"})
        d = entry.to_dict()
        assert d["sequence"] == 0
        assert d["op_type"] == "fs_mv"
        assert d["params"] == {"src": "/a", "dst": "/b"}
        assert d["completed"] is False

    def test_from_dict(self):
        data = {"sequence": 1, "op_type": "fs_rm", "params": {"uri": "/x"}, "completed": True}
        entry = UndoEntry.from_dict(data)
        assert entry.sequence == 1
        assert entry.op_type == "fs_rm"
        assert entry.completed is True

    def test_roundtrip(self):
        entry = UndoEntry(
            sequence=5, op_type="vectordb_upsert", params={"record_id": "r1"}, completed=True
        )
        restored = UndoEntry.from_dict(entry.to_dict())
        assert restored.sequence == entry.sequence
        assert restored.op_type == entry.op_type
        assert restored.params == entry.params
        assert restored.completed == entry.completed


class TestExecuteRollback:
    """Integration tests for execute_rollback using real AGFS and VectorDB backends."""

    async def test_rollback_fs_mv(self, agfs_client, test_dir):
        src = f"{test_dir}/src"
        dst = f"{test_dir}/dst"
        _mkdir_ok(agfs_client, src)
        agfs_client.write(f"{src}/data.txt", b"hello")

        # Forward: mv src → dst
        agfs_client.mv(src, dst)
        assert not file_exists(agfs_client, src)
        assert file_exists(agfs_client, dst)

        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="fs_mv",
                params={"src": src, "dst": dst},
                completed=True,
            ),
        ]
        await execute_rollback(undo_log, agfs_client)

        # src restored, dst gone
        assert file_exists(agfs_client, src)
        assert not file_exists(agfs_client, dst)

    async def test_rollback_fs_rm_skipped(self, agfs_client, test_dir):
        path = f"{test_dir}/will-not-delete"
        _mkdir_ok(agfs_client, path)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_rm", params={"uri": path}, completed=True),
        ]
        await execute_rollback(undo_log, agfs_client)

        # fs_rm rollback is a no-op; directory still exists
        assert file_exists(agfs_client, path)

    async def test_rollback_fs_mkdir(self, agfs_client, test_dir):
        new_dir = f"{test_dir}/created"
        _mkdir_ok(agfs_client, new_dir)
        assert file_exists(agfs_client, new_dir)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_mkdir", params={"uri": new_dir}, completed=True),
        ]
        await execute_rollback(undo_log, agfs_client)

        assert not file_exists(agfs_client, new_dir)

    async def test_rollback_fs_write_new(self, agfs_client, test_dir):
        file_path = f"{test_dir}/new-file.txt"
        agfs_client.write(file_path, b"content")
        assert file_exists(agfs_client, file_path)

        undo_log = [
            UndoEntry(
                sequence=0, op_type="fs_write_new", params={"uri": file_path}, completed=True
            ),
        ]
        await execute_rollback(undo_log, agfs_client)

        assert not file_exists(agfs_client, file_path)

    async def test_rollback_reverse_order(self, agfs_client, test_dir):
        """mkdir parent + child → rollback → both removed in reverse order."""
        parent = f"{test_dir}/parent"
        child = f"{test_dir}/parent/child"
        _mkdir_ok(agfs_client, parent)
        _mkdir_ok(agfs_client, child)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_mkdir", params={"uri": parent}, completed=True),
            UndoEntry(sequence=1, op_type="fs_mkdir", params={"uri": child}, completed=True),
        ]
        await execute_rollback(undo_log, agfs_client)

        # child removed first (seq=1), then parent (seq=0)
        assert not file_exists(agfs_client, child)
        assert not file_exists(agfs_client, parent)

    async def test_rollback_skips_incomplete(self, agfs_client, test_dir):
        new_dir = f"{test_dir}/incomplete"
        _mkdir_ok(agfs_client, new_dir)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_mkdir", params={"uri": new_dir}, completed=False),
        ]
        await execute_rollback(undo_log, agfs_client)

        # completed=False → not rolled back
        assert file_exists(agfs_client, new_dir)

    async def test_rollback_best_effort(self, agfs_client, test_dir):
        """A failing rollback entry should not prevent others from running."""
        real_dir = f"{test_dir}/real-dir"
        _mkdir_ok(agfs_client, real_dir)

        src = f"{test_dir}/be-src"
        dst = f"{test_dir}/be-dst"
        _mkdir_ok(agfs_client, dst)

        undo_log = [
            # seq=0: fs_mv rollback will succeed
            UndoEntry(sequence=0, op_type="fs_mv", params={"src": src, "dst": dst}, completed=True),
            # seq=1: fs_mkdir rollback will fail (rm on non-empty or non-existent path)
            UndoEntry(
                sequence=1,
                op_type="fs_mkdir",
                params={"uri": f"{test_dir}/nonexistent-dir-xyz"},
                completed=True,
            ),
        ]
        # Should not raise
        await execute_rollback(undo_log, agfs_client)

        # seq=0 mv rollback should have executed (dst → src)
        assert file_exists(agfs_client, src)

    async def test_rollback_vectordb_upsert(self, agfs_client, vector_store, request_ctx):
        """Real upsert → rollback → record deleted."""
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "uri": f"viking://resources/test-upsert-{record_id}.md",
            "parent_uri": "viking://resources/",
            "account_id": "default",
            "context_type": "resource",
            "level": 2,
            "vector": [0.1] * VECTOR_DIM,
            "name": "test",
            "description": "test record",
            "abstract": "test",
        }
        await vector_store.upsert(record, ctx=request_ctx)

        # Confirm it exists
        results = await vector_store.get([record_id], ctx=request_ctx)
        assert len(results) == 1

        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="vectordb_upsert",
                params={
                    "record_id": record_id,
                    "_ctx_account_id": "default",
                    "_ctx_user_id": "test_user",
                    "_ctx_role": "root",
                },
                completed=True,
            ),
        ]
        await execute_rollback(undo_log, agfs_client, vector_store=vector_store)

        results = await vector_store.get([record_id], ctx=request_ctx)
        assert len(results) == 0

    async def test_rollback_vectordb_update_uri(self, agfs_client, vector_store, request_ctx):
        """Real upsert → update_uri_mapping → rollback → URI restored."""
        record_id = str(uuid.uuid4())
        old_uri = f"viking://resources/old-{record_id}.md"
        new_uri = f"viking://resources/new-{record_id}.md"
        record = {
            "id": record_id,
            "uri": old_uri,
            "parent_uri": "viking://resources/",
            "account_id": "default",
            "context_type": "resource",
            "level": 2,
            "vector": [0.2] * VECTOR_DIM,
            "name": "test",
            "description": "test",
            "abstract": "test",
        }
        await vector_store.upsert(record, ctx=request_ctx)

        # Forward: update URI mapping
        await vector_store.update_uri_mapping(
            ctx=request_ctx,
            uri=old_uri,
            new_uri=new_uri,
            new_parent_uri="viking://resources/",
        )

        # Verify forward operation
        result = await vector_store.fetch_by_uri(new_uri, ctx=request_ctx)
        assert result is not None

        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="vectordb_update_uri",
                params={
                    "old_uri": old_uri,
                    "new_uri": new_uri,
                    "old_parent_uri": "viking://resources/",
                    "_ctx_account_id": "default",
                    "_ctx_user_id": "test_user",
                    "_ctx_role": "root",
                },
                completed=True,
            ),
        ]
        await execute_rollback(undo_log, agfs_client, vector_store=vector_store)

        # URI should be restored to old_uri
        result = await vector_store.fetch_by_uri(old_uri, ctx=request_ctx)
        assert result is not None
