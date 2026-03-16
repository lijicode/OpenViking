# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests: multi-step rollback covering FS + VectorDB coordination."""

import uuid

from openviking.storage.transaction.undo import UndoEntry, execute_rollback

from .conftest import VECTOR_DIM, _mkdir_ok, file_exists


class TestRmRollback:
    def test_fs_rm_not_reversible(self, agfs_client, test_dir):
        """fs_rm is intentionally irreversible: even completed=True is a no-op."""
        path = f"{test_dir}/rm-target"
        _mkdir_ok(agfs_client, path)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_rm", params={"uri": path}, completed=True),
        ]
        execute_rollback(undo_log, agfs_client)

        # Directory still exists — fs_rm rollback does nothing
        assert file_exists(agfs_client, path)


class TestMvRollback:
    def test_mv_reversed_on_rollback(self, agfs_client, test_dir):
        """Real mv → rollback → content back at original location."""
        src = f"{test_dir}/mv-src"
        dst = f"{test_dir}/mv-dst"
        _mkdir_ok(agfs_client, src)
        agfs_client.write(f"{src}/payload.txt", b"important data")

        # Forward mv
        agfs_client.mv(src, dst)
        assert not file_exists(agfs_client, src)
        content = agfs_client.cat(f"{dst}/payload.txt")
        assert content == b"important data"

        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="fs_mv",
                params={"src": src, "dst": dst},
                completed=True,
            ),
        ]
        execute_rollback(undo_log, agfs_client)

        assert file_exists(agfs_client, src)
        restored = agfs_client.cat(f"{src}/payload.txt")
        assert restored == b"important data"


class TestRecoverAll:
    def test_recover_all_reverses_incomplete(self, agfs_client, test_dir):
        """recover_all=True also reverses entries with completed=False."""
        new_dir = f"{test_dir}/recover-all-dir"
        _mkdir_ok(agfs_client, new_dir)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_mkdir", params={"uri": new_dir}, completed=False),
        ]
        execute_rollback(undo_log, agfs_client, recover_all=True)

        assert not file_exists(agfs_client, new_dir)

    def test_recover_all_false_skips_incomplete(self, agfs_client, test_dir):
        """recover_all=False skips entries with completed=False."""
        new_dir = f"{test_dir}/skip-incomplete"
        _mkdir_ok(agfs_client, new_dir)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_mkdir", params={"uri": new_dir}, completed=False),
        ]
        execute_rollback(undo_log, agfs_client, recover_all=False)

        assert file_exists(agfs_client, new_dir)


class TestMultiStepRollback:
    def test_reverse_order_nested_dirs(self, agfs_client, test_dir):
        """parent + child → rollback reverses in reverse sequence order."""
        parent = f"{test_dir}/multi-parent"
        child = f"{test_dir}/multi-parent/child"
        _mkdir_ok(agfs_client, parent)
        _mkdir_ok(agfs_client, child)

        undo_log = [
            UndoEntry(sequence=0, op_type="fs_mkdir", params={"uri": parent}, completed=True),
            UndoEntry(sequence=1, op_type="fs_mkdir", params={"uri": child}, completed=True),
        ]
        execute_rollback(undo_log, agfs_client)

        assert not file_exists(agfs_client, child)
        assert not file_exists(agfs_client, parent)

    def test_write_new_rollback(self, agfs_client, test_dir):
        """New file → rollback → file deleted."""
        file_path = f"{test_dir}/new-file.txt"
        agfs_client.write(file_path, b"new content")
        assert file_exists(agfs_client, file_path)

        undo_log = [
            UndoEntry(
                sequence=0, op_type="fs_write_new", params={"uri": file_path}, completed=True
            ),
        ]
        execute_rollback(undo_log, agfs_client)

        assert not file_exists(agfs_client, file_path)

    def test_best_effort_continues(self, agfs_client, test_dir):
        """If one step fails, subsequent steps still execute."""
        real_dir = f"{test_dir}/best-effort-real"
        _mkdir_ok(agfs_client, real_dir)

        undo_log = [
            # seq=0: mkdir rollback on real dir → should succeed
            UndoEntry(sequence=0, op_type="fs_mkdir", params={"uri": real_dir}, completed=True),
            # seq=1: mkdir rollback on nonexistent dir → fails silently
            UndoEntry(
                sequence=1,
                op_type="fs_mkdir",
                params={"uri": f"{test_dir}/no-such-dir-{uuid.uuid4().hex}"},
                completed=True,
            ),
        ]
        execute_rollback(undo_log, agfs_client)

        # seq=0 still executed despite seq=1 failure (reversed order: 1 runs first, then 0)
        assert not file_exists(agfs_client, real_dir)

    def test_unknown_op_type_no_crash(self, agfs_client, test_dir):
        """Unknown op_type is logged but doesn't raise."""
        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="some_future_op",
                params={"foo": "bar"},
                completed=True,
            ),
        ]
        # Should not raise
        execute_rollback(undo_log, agfs_client)


class TestVectorDBRollback:
    async def test_vectordb_delete_rollback_restores(self, agfs_client, vector_store, request_ctx):
        """upsert → delete → rollback(vectordb_delete) → record restored."""
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "uri": f"viking://resources/del-restore-{record_id}.md",
            "parent_uri": "viking://resources/",
            "account_id": "default",
            "context_type": "resource",
            "level": 2,
            "vector": [0.3] * VECTOR_DIM,
            "name": "del-restore",
            "description": "test",
            "abstract": "test",
        }
        await vector_store.upsert(record, ctx=request_ctx)

        # Snapshot before delete
        snapshot = await vector_store.get([record_id], ctx=request_ctx)
        assert len(snapshot) == 1

        # Forward: delete
        await vector_store.delete([record_id], ctx=request_ctx)
        assert len(await vector_store.get([record_id], ctx=request_ctx)) == 0

        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="vectordb_delete",
                params={
                    "uris": [record["uri"]],
                    "records_snapshot": snapshot,
                    "_ctx_account_id": "default",
                    "_ctx_user_id": "test_user",
                    "_ctx_role": "root",
                },
                completed=True,
            ),
        ]
        execute_rollback(undo_log, agfs_client, vector_store=vector_store)

        results = await vector_store.get([record_id], ctx=request_ctx)
        assert len(results) == 1

    async def test_vectordb_delete_multi_record(self, agfs_client, vector_store, request_ctx):
        """3 records in snapshot → rollback → all restored."""
        records = []
        for i in range(3):
            rid = str(uuid.uuid4())
            rec = {
                "id": rid,
                "uri": f"viking://resources/multi-{rid}.md",
                "parent_uri": "viking://resources/",
                "account_id": "default",
                "context_type": "resource",
                "level": 2,
                "vector": [0.1 * (i + 1)] * VECTOR_DIM,
                "name": f"multi-{i}",
                "description": "test",
                "abstract": "test",
            }
            await vector_store.upsert(rec, ctx=request_ctx)
            records.append(rec)

        ids = [r["id"] for r in records]
        snapshot = await vector_store.get(ids, ctx=request_ctx)
        assert len(snapshot) == 3

        # Delete all
        await vector_store.delete(ids, ctx=request_ctx)
        assert len(await vector_store.get(ids, ctx=request_ctx)) == 0

        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="vectordb_delete",
                params={
                    "uris": [r["uri"] for r in records],
                    "records_snapshot": snapshot,
                    "_ctx_account_id": "default",
                    "_ctx_user_id": "test_user",
                    "_ctx_role": "root",
                },
                completed=True,
            ),
        ]
        execute_rollback(undo_log, agfs_client, vector_store=vector_store)

        results = await vector_store.get(ids, ctx=request_ctx)
        assert len(results) == 3

    async def test_vectordb_delete_empty_snapshot(self, agfs_client, vector_store, request_ctx):
        """Empty snapshot → no-op, no error."""
        undo_log = [
            UndoEntry(
                sequence=0,
                op_type="vectordb_delete",
                params={
                    "uris": [],
                    "records_snapshot": [],
                    "_ctx_account_id": "default",
                    "_ctx_user_id": "test_user",
                    "_ctx_role": "root",
                },
                completed=True,
            ),
        ]
        # Should not raise
        execute_rollback(undo_log, agfs_client, vector_store=vector_store)

    async def test_vectordb_upsert_rollback_deletes(self, agfs_client, vector_store, request_ctx):
        """upsert → rollback(vectordb_upsert) → record deleted."""
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "uri": f"viking://resources/upsert-del-{record_id}.md",
            "parent_uri": "viking://resources/",
            "account_id": "default",
            "context_type": "resource",
            "level": 2,
            "vector": [0.4] * VECTOR_DIM,
            "name": "upsert-del",
            "description": "test",
            "abstract": "test",
        }
        await vector_store.upsert(record, ctx=request_ctx)
        assert len(await vector_store.get([record_id], ctx=request_ctx)) == 1

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
        execute_rollback(undo_log, agfs_client, vector_store=vector_store)

        results = await vector_store.get([record_id], ctx=request_ctx)
        assert len(results) == 0
