# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Undo log and rollback executor for transaction management.

Records operations performed within a transaction so they can be reversed
on rollback. Each UndoEntry captures one atomic sub-operation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


def _reconstruct_ctx(params: Dict[str, Any]) -> Optional[Any]:
    """Reconstruct a RequestContext from serialized _ctx_* fields in undo params.

    Returns None if the required fields are missing.
    """
    account_id = params.get("_ctx_account_id")
    user_id = params.get("_ctx_user_id")
    agent_id = params.get("_ctx_agent_id")
    role_value = params.get("_ctx_role")
    if account_id is None or user_id is None:
        return None
    try:
        from openviking.server.identity import RequestContext, Role
        from openviking_cli.session.user_id import UserIdentifier

        role = Role(role_value) if role_value in {r.value for r in Role} else Role.ROOT
        user = UserIdentifier(account_id, user_id, agent_id or "default")
        return RequestContext(user=user, role=role)
    except Exception as e:
        logger.warning(f"[Rollback] Failed to reconstruct ctx: {e}")
        return None


@dataclass
class UndoEntry:
    """A single undo log entry representing one reversible sub-operation.

    Attributes:
        sequence: Monotonically increasing index within the transaction.
        op_type: Operation type (fs_mv, fs_rm, fs_mkdir, fs_write_new,
                 vectordb_upsert, vectordb_delete, vectordb_update_uri).
        params: Parameters needed to reverse the operation.
        completed: Whether the forward operation completed successfully.
    """

    sequence: int
    op_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "op_type": self.op_type,
            "params": self.params,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UndoEntry":
        return cls(
            sequence=data.get("sequence", 0),
            op_type=data.get("op_type", ""),
            params=data.get("params", {}),
            completed=data.get("completed", False),
        )


async def execute_rollback(
    undo_log: List[UndoEntry],
    agfs: Any,
    vector_store: Optional[Any] = None,
    ctx: Optional[Any] = None,
    recover_all: bool = False,
) -> None:
    """Execute rollback by reversing operations in reverse order.

    Best-effort: each step is wrapped in try-except so a single failure
    does not prevent subsequent undo steps from running.

    Args:
        undo_log: List of undo entries to process.
        agfs: AGFS client for filesystem operations.
        vector_store: Optional vector store client.
        ctx: Optional request context.
        recover_all: If True, also attempt to reverse entries that were not
            marked completed (used during crash recovery to clean up partial
            operations such as a directory mv that only half-finished).
    """
    if recover_all:
        entries = list(undo_log)
    else:
        entries = [e for e in undo_log if e.completed]
    entries.sort(key=lambda e: e.sequence, reverse=True)

    for entry in entries:
        try:
            await _rollback_entry(entry, agfs, vector_store, ctx)
            logger.info(f"[Rollback] Reversed {entry.op_type} seq={entry.sequence}")
        except Exception as e:
            logger.warning(
                f"[Rollback] Failed to reverse {entry.op_type} seq={entry.sequence}: {e}"
            )


async def _rollback_entry(
    entry: UndoEntry,
    agfs: Any,
    vector_store: Optional[Any],
    ctx: Optional[Any],
) -> None:
    """Dispatch rollback for a single undo entry."""
    op = entry.op_type
    params = entry.params

    if op == "fs_mv":
        agfs.mv(params["dst"], params["src"])

    elif op == "fs_rm":
        logger.debug("[Rollback] fs_rm is not reversible, skipping")

    elif op == "fs_mkdir":
        try:
            agfs.rm(params["uri"])
        except Exception:
            pass

    elif op == "fs_write_new":
        try:
            agfs.rm(params["uri"], recursive=True)
        except Exception:
            pass

    elif op == "vectordb_upsert":
        if vector_store:
            record_id = params.get("record_id")
            if record_id:
                restored_ctx = _reconstruct_ctx(params)
                if restored_ctx:
                    await vector_store.delete([record_id], ctx=restored_ctx)
                else:
                    logger.warning("[Rollback] vectordb_upsert: cannot reconstruct ctx, skipping")

    elif op == "vectordb_delete":
        if vector_store:
            restored_ctx = _reconstruct_ctx(params)
            if restored_ctx is None:
                logger.warning("[Rollback] vectordb_delete: cannot reconstruct ctx, skipping")
            else:
                records_snapshot = params.get("records_snapshot", [])
                for record in records_snapshot:
                    try:
                        await vector_store.upsert(record, ctx=restored_ctx)
                    except Exception as e:
                        logger.warning(f"[Rollback] Failed to restore vector record: {e}")

    elif op == "vectordb_update_uri":
        if vector_store:
            restored_ctx = _reconstruct_ctx(params)
            if restored_ctx is None:
                logger.warning("[Rollback] vectordb_update_uri: cannot reconstruct ctx, skipping")
            else:
                await vector_store.update_uri_mapping(
                    ctx=restored_ctx,
                    uri=params["new_uri"],
                    new_uri=params["old_uri"],
                    new_parent_uri=params.get("old_parent_uri", ""),
                )

    else:
        logger.warning(f"[Rollback] Unknown op_type: {op}")
