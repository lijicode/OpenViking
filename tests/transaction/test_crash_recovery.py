# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Integration test: crash recovery from journal using real AGFS and VectorDB backends."""

import uuid
from unittest.mock import AsyncMock, patch

from openviking.storage.transaction.journal import TransactionJournal
from openviking.storage.transaction.transaction_manager import TransactionManager
from openviking.storage.transaction.transaction_record import (
    TransactionRecord,
    TransactionStatus,
)
from openviking.storage.transaction.undo import UndoEntry

from .conftest import VECTOR_DIM, _mkdir_ok, file_exists, make_lock_file


def _write_journal(journal, record):
    """Write a TransactionRecord to real journal storage."""
    journal.write(record.to_journal())


class TestCrashRecovery:
    """
    Core technique: simulate crash recovery.

    1. Create real FS state via agfs_client
    2. Build TransactionRecord, write to real journal
    3. Create fresh TransactionManager (simulates process restart)
    4. Call manager._recover_pending_transactions()
    5. Verify final state via agfs_client.stat()/cat() and vector_store.get()
    """

    async def test_recover_commit_no_rollback(self, agfs_client, vector_store, test_dir):
        """COMMIT status → committed files NOT rolled back, journal cleaned up."""
        # Create a file that was part of a committed transaction
        committed_file = f"{test_dir}/committed.txt"
        agfs_client.write(committed_file, b"committed data")

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-commit-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.COMMIT,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_write_new",
                    params={"uri": committed_file},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        # New manager (simulates restart)
        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        # File should still exist (no rollback for committed tx)
        assert file_exists(agfs_client, committed_file)
        # Journal should be cleaned up
        assert tx_id not in journal.list_all()

    async def test_recover_commit_replays_post_actions(self, agfs_client, vector_store, test_dir):
        """COMMIT + post_actions → replay post_actions."""
        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-post-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.COMMIT,
            locks=[],
            undo_log=[],
            post_actions=[
                {
                    "type": "enqueue_semantic",
                    "params": {
                        "uri": "viking://test-post",
                        "context_type": "resource",
                        "account_id": "acc",
                    },
                }
            ],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)

        with patch.object(manager, "_execute_post_actions", new_callable=AsyncMock) as mock_post:
            await manager._recover_pending_transactions()

        mock_post.assert_called_once()
        assert tx_id not in journal.list_all()

    async def test_recover_exec_rollback_fs_mv(self, agfs_client, vector_store, test_dir):
        """EXEC status with fs_mv → recovery rolls back → file moved back."""
        src = f"{test_dir}/exec-mv-src"
        dst = f"{test_dir}/exec-mv-dst"
        _mkdir_ok(agfs_client, src)
        agfs_client.write(f"{src}/data.txt", b"mv-data")

        # Simulate: forward mv happened, then crash
        agfs_client.mv(src, dst)
        assert not file_exists(agfs_client, src)

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-exec-mv-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.EXEC,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_mv",
                    params={"src": src, "dst": dst},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert file_exists(agfs_client, src)
        assert not file_exists(agfs_client, dst)
        assert tx_id not in journal.list_all()

    async def test_recover_exec_rollback_fs_mkdir(self, agfs_client, vector_store, test_dir):
        """EXEC with fs_mkdir → recovery → directory removed."""
        new_dir = f"{test_dir}/exec-mkdir"
        _mkdir_ok(agfs_client, new_dir)

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-exec-mkdir-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.EXEC,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_mkdir",
                    params={"uri": new_dir},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, new_dir)
        assert tx_id not in journal.list_all()

    async def test_recover_exec_rollback_fs_write_new(self, agfs_client, vector_store, test_dir):
        """EXEC with fs_write_new → recovery → file removed."""
        file_path = f"{test_dir}/exec-write.txt"
        agfs_client.write(file_path, b"to-be-rolled-back")

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-exec-write-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.EXEC,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_write_new",
                    params={"uri": file_path},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, file_path)
        assert tx_id not in journal.list_all()

    async def test_recover_exec_rollback_vectordb_upsert(
        self, agfs_client, vector_store, request_ctx, test_dir
    ):
        """EXEC with vectordb_upsert → recovery → record deleted from VectorDB."""
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "uri": f"viking://resources/crash-upsert-{record_id}.md",
            "parent_uri": "viking://resources/",
            "account_id": "default",
            "context_type": "resource",
            "level": 2,
            "vector": [0.5] * VECTOR_DIM,
            "name": "crash-upsert",
            "description": "test",
            "abstract": "test",
        }
        await vector_store.upsert(record, ctx=request_ctx)
        assert len(await vector_store.get([record_id], ctx=request_ctx)) == 1

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-exec-vdb-{uuid.uuid4().hex[:8]}"
        tx_record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.EXEC,
            locks=[],
            undo_log=[
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
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, tx_record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        results = await vector_store.get([record_id], ctx=request_ctx)
        assert len(results) == 0
        assert tx_id not in journal.list_all()

    async def test_recover_fail_triggers_rollback(self, agfs_client, vector_store, test_dir):
        """FAIL status → also triggers rollback."""
        new_dir = f"{test_dir}/fail-dir"
        _mkdir_ok(agfs_client, new_dir)

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-fail-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.FAIL,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_mkdir",
                    params={"uri": new_dir},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, new_dir)
        assert tx_id not in journal.list_all()

    async def test_recover_releasing_triggers_rollback(self, agfs_client, vector_store, test_dir):
        """RELEASING status → rollback + lock cleanup."""
        new_dir = f"{test_dir}/releasing-dir"
        _mkdir_ok(agfs_client, new_dir)

        lock_path = make_lock_file(agfs_client, test_dir, "tx-releasing-placeholder", "S")

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-releasing-{uuid.uuid4().hex[:8]}"
        # Rewrite lock with correct tx_id
        lock_path = make_lock_file(agfs_client, test_dir, tx_id, "S")

        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.RELEASING,
            locks=[lock_path],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_mkdir",
                    params={"uri": new_dir},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, new_dir)
        assert not file_exists(agfs_client, lock_path)
        assert tx_id not in journal.list_all()

    async def test_recover_exec_includes_incomplete(self, agfs_client, vector_store, test_dir):
        """EXEC recovery uses recover_all=True → also reverses incomplete entries."""
        new_dir = f"{test_dir}/exec-incomplete"
        _mkdir_ok(agfs_client, new_dir)

        journal = TransactionJournal(agfs_client)
        tx_id = f"tx-exec-inc-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.EXEC,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_mkdir",
                    params={"uri": new_dir},
                    completed=False,  # incomplete, but recover_all=True reverses it
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, new_dir)
        assert tx_id not in journal.list_all()

    async def test_recover_init_cleans_locks(self, agfs_client, vector_store, test_dir):
        """INIT status → no rollback, just lock cleanup + journal delete."""
        lock_dir = f"{test_dir}/init-lock-dir"
        _mkdir_ok(agfs_client, lock_dir)

        tx_id = f"tx-init-{uuid.uuid4().hex[:8]}"
        lock_path = make_lock_file(agfs_client, lock_dir, tx_id, "P")

        journal = TransactionJournal(agfs_client)
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.INIT,
            locks=[lock_path],
            undo_log=[],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, lock_path)
        assert tx_id not in journal.list_all()

    async def test_recover_acquire_cleans_locks(self, agfs_client, vector_store, test_dir):
        """ACQUIRE status → same as INIT, clean up only."""
        lock_dir = f"{test_dir}/acquire-lock-dir"
        _mkdir_ok(agfs_client, lock_dir)

        tx_id = f"tx-acq-{uuid.uuid4().hex[:8]}"
        lock_path = make_lock_file(agfs_client, lock_dir, tx_id, "P")

        journal = TransactionJournal(agfs_client)
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.ACQUIRE,
            locks=[lock_path],
            undo_log=[],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, lock_path)
        assert tx_id not in journal.list_all()

    async def test_recover_init_orphan_lock_via_init_info(
        self, agfs_client, vector_store, test_dir
    ):
        """INIT with empty locks but init_info.lock_paths → clean orphan lock owned by tx."""
        orphan_dir = f"{test_dir}/orphan-dir"
        _mkdir_ok(agfs_client, orphan_dir)

        tx_id = f"tx-orphan-{uuid.uuid4().hex[:8]}"
        lock_path = make_lock_file(agfs_client, orphan_dir, tx_id, "S")

        journal = TransactionJournal(agfs_client)
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.INIT,
            locks=[],  # Empty — crash happened before journal recorded locks
            init_info={
                "operation": "rm",
                "lock_paths": [orphan_dir],
                "lock_mode": "subtree",
            },
            undo_log=[],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, lock_path)
        assert tx_id not in journal.list_all()

    async def test_recover_init_orphan_lock_other_owner(self, agfs_client, vector_store, test_dir):
        """INIT with orphan lock owned by different tx → not removed."""
        orphan_dir = f"{test_dir}/orphan-other"
        _mkdir_ok(agfs_client, orphan_dir)

        other_tx_id = f"tx-OTHER-{uuid.uuid4().hex[:8]}"
        lock_path = make_lock_file(agfs_client, orphan_dir, other_tx_id, "S")

        tx_id = f"tx-innocent-{uuid.uuid4().hex[:8]}"
        journal = TransactionJournal(agfs_client)
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.INIT,
            locks=[],
            init_info={
                "operation": "rm",
                "lock_paths": [orphan_dir],
                "lock_mode": "subtree",
            },
            undo_log=[],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        # Lock file should still exist — owned by different tx
        assert file_exists(agfs_client, lock_path)
        assert tx_id not in journal.list_all()

    async def test_recover_mv_orphan_both_paths(self, agfs_client, vector_store, test_dir):
        """INIT mv operation → check both lock_paths and mv_dst_path for orphan locks."""
        src_dir = f"{test_dir}/mv-orphan-src"
        dst_dir = f"{test_dir}/mv-orphan-dst"
        _mkdir_ok(agfs_client, src_dir)
        _mkdir_ok(agfs_client, dst_dir)

        tx_id = f"tx-mv-orphan-{uuid.uuid4().hex[:8]}"
        src_lock = make_lock_file(agfs_client, src_dir, tx_id, "S")
        dst_lock = make_lock_file(agfs_client, dst_dir, tx_id, "P")

        journal = TransactionJournal(agfs_client)
        record = TransactionRecord(
            id=tx_id,
            status=TransactionStatus.INIT,
            locks=[],
            init_info={
                "operation": "mv",
                "lock_paths": [src_dir],
                "lock_mode": "mv",
                "mv_dst_path": dst_dir,
            },
            undo_log=[],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        # Both orphan locks should be cleaned up
        assert not file_exists(agfs_client, src_lock)
        assert not file_exists(agfs_client, dst_lock)
        assert tx_id not in journal.list_all()

    async def test_recover_multiple_transactions(self, agfs_client, vector_store, test_dir):
        """Multiple journal entries are all recovered."""
        dir_a = f"{test_dir}/multi-tx-a"
        _mkdir_ok(agfs_client, dir_a)

        journal = TransactionJournal(agfs_client)

        # tx-a: EXEC with mkdir → should rollback
        tx_a = f"tx-multi-a-{uuid.uuid4().hex[:8]}"
        record_a = TransactionRecord(
            id=tx_a,
            status=TransactionStatus.EXEC,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_mkdir",
                    params={"uri": dir_a},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record_a)

        # tx-b: COMMIT → no rollback, just cleanup
        tx_b = f"tx-multi-b-{uuid.uuid4().hex[:8]}"
        record_b = TransactionRecord(
            id=tx_b,
            status=TransactionStatus.COMMIT,
            locks=[],
            undo_log=[],
            post_actions=[],
        )
        _write_journal(journal, record_b)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        assert not file_exists(agfs_client, dir_a)  # rolled back
        assert tx_a not in journal.list_all()
        assert tx_b not in journal.list_all()

    async def test_recover_corrupted_journal_skips(self, agfs_client, vector_store, test_dir):
        """Corrupted journal entry → skipped, others still processed."""
        journal = TransactionJournal(agfs_client)

        # Write a corrupted journal entry (invalid JSON)
        bad_tx_id = f"tx-bad-{uuid.uuid4().hex[:8]}"
        _mkdir_ok(agfs_client, "/local/_system")
        _mkdir_ok(agfs_client, "/local/_system/transactions")
        bad_dir = f"/local/_system/transactions/{bad_tx_id}"
        _mkdir_ok(agfs_client, bad_dir)
        agfs_client.write(f"{bad_dir}/journal.json", b"NOT VALID JSON {{{{")

        # Write a good journal entry
        good_dir = f"{test_dir}/good-recovery"
        _mkdir_ok(agfs_client, good_dir)

        good_tx_id = f"tx-good-{uuid.uuid4().hex[:8]}"
        record = TransactionRecord(
            id=good_tx_id,
            status=TransactionStatus.EXEC,
            locks=[],
            undo_log=[
                UndoEntry(
                    sequence=0,
                    op_type="fs_mkdir",
                    params={"uri": good_dir},
                    completed=True,
                )
            ],
            post_actions=[],
        )
        _write_journal(journal, record)

        manager = TransactionManager(agfs_client=agfs_client, vector_store=vector_store)
        await manager._recover_pending_transactions()

        # Good tx should still be recovered
        assert not file_exists(agfs_client, good_dir)
        assert good_tx_id not in journal.list_all()
