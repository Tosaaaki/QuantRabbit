from __future__ import annotations

from typing import Any

import pytest

from analytics.bq_exporter import BigQueryExporter


class _FakeBigQueryClient:
    def __init__(self, *, errors_by_call: list[list[dict[str, Any]]] | None = None) -> None:
        self.project = "quantrabbit"
        self.calls: list[dict[str, Any]] = []
        self._errors_by_call = errors_by_call or []

    def insert_rows_json(
        self,
        _table_ref: str,
        rows: list[dict[str, Any]],
        *,
        row_ids: list[str],
        retry: Any,
        timeout: float | None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "rows": list(rows),
                "row_ids": list(row_ids),
                "retry": retry,
                "timeout": timeout,
            }
        )
        idx = len(self.calls) - 1
        if idx < len(self._errors_by_call):
            return self._errors_by_call[idx]
        return []


def _build_exporter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: list[dict[str, Any]],
    fake_client: _FakeBigQueryClient,
) -> tuple[BigQueryExporter, list[str]]:
    monkeypatch.setattr(BigQueryExporter, "_ensure_dataset", lambda self: None)
    monkeypatch.setattr(BigQueryExporter, "_ensure_table", lambda self: None)

    exporter = BigQueryExporter(
        sqlite_path="logs/trades.db",
        state_path="logs/bq_sync_state.test.json",
        project_id="quantrabbit",
    )
    exporter.client = fake_client
    exporter.batch_size = 2
    exporter.insert_timeout_sec = 7.0
    exporter.insert_retry = object()

    monkeypatch.setattr(exporter, "_fetch_trades", lambda _limit: list(rows))
    cursor_updates: list[str] = []
    monkeypatch.setattr(exporter, "_set_last_cursor", lambda value: cursor_updates.append(str(value)))
    return exporter, cursor_updates


def test_export_batches_rows_and_updates_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"ticket_id": "t1", "updated_at": "2026-02-20T09:00:01+00:00", "transaction_id": 1},
        {"ticket_id": "t2", "updated_at": "2026-02-20T09:00:02+00:00", "transaction_id": 2},
        {"ticket_id": "t3", "updated_at": "2026-02-20T09:00:03+00:00", "transaction_id": 3},
        {"ticket_id": "t4", "updated_at": "2026-02-20T09:00:04+00:00"},
        {"ticket_id": "t5", "updated_at": "2026-02-20T09:00:05+00:00", "transaction_id": 5},
    ]
    fake_client = _FakeBigQueryClient()
    exporter, cursor_updates = _build_exporter(monkeypatch, rows=rows, fake_client=fake_client)

    stats = exporter.export(limit=500)

    assert stats.exported == 5
    assert stats.last_updated_at == "2026-02-20T09:00:05+00:00"
    assert cursor_updates == ["2026-02-20T09:00:05+00:00"]
    assert [len(c["rows"]) for c in fake_client.calls] == [2, 2, 1]
    assert all(c["timeout"] == 7.0 for c in fake_client.calls)
    assert fake_client.calls[0]["row_ids"] == [
        "t1:2026-02-20T09:00:01+00:00:1",
        "t2:2026-02-20T09:00:02+00:00:2",
    ]
    assert fake_client.calls[1]["row_ids"] == [
        "t3:2026-02-20T09:00:03+00:00:3",
        "t4:2026-02-20T09:00:04+00:00",
    ]


def test_export_raises_on_chunk_error_and_does_not_advance_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"ticket_id": "t1", "updated_at": "2026-02-20T09:00:01+00:00", "transaction_id": 1},
        {"ticket_id": "t2", "updated_at": "2026-02-20T09:00:02+00:00", "transaction_id": 2},
        {"ticket_id": "t3", "updated_at": "2026-02-20T09:00:03+00:00", "transaction_id": 3},
    ]
    fake_client = _FakeBigQueryClient(
        errors_by_call=[
            [],
            [{"index": 0, "errors": [{"reason": "backendError"}]}],
        ]
    )
    exporter, cursor_updates = _build_exporter(monkeypatch, rows=rows, fake_client=fake_client)

    with pytest.raises(RuntimeError, match="chunk_offset=2"):
        exporter.export(limit=500)

    assert [len(c["rows"]) for c in fake_client.calls] == [2, 1]
    assert cursor_updates == []
