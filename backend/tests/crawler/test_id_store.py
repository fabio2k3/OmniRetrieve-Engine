"""Tests de IdStore — IDs compuestos, CSV, persistencia, thread safety."""
from __future__ import annotations
import csv
import threading
import pytest


def test_add_composite_ids(tmp_path):
    from backend.crawler.id_store import IdStore
    store = IdStore(tmp_path / "ids.csv")
    added = store.add_ids(["arxiv:001", "arxiv:002", "fake:abc"])
    assert added == 3
    assert store.total == 3


def test_deduplication(tmp_path):
    from backend.crawler.id_store import IdStore
    store = IdStore(tmp_path / "ids.csv")
    store.add_ids(["arxiv:001", "arxiv:002"])
    added = store.add_ids(["arxiv:001", "arxiv:003"])
    assert added == 1
    assert store.total == 3


def test_get_pending_batch_returns_composite_ids(tmp_path):
    from backend.crawler.id_store import IdStore
    store = IdStore(tmp_path / "ids.csv")
    store.add_ids(["arxiv:001", "fake:doc1", "arxiv:002"])
    batch = store.get_pending_batch(2)
    assert len(batch) == 2
    assert all(":" in doc_id for doc_id in batch)


def test_mark_downloaded(tmp_path):
    from backend.crawler.id_store import IdStore
    store = IdStore(tmp_path / "ids.csv")
    store.add_ids(["arxiv:001", "arxiv:002", "arxiv:003"])
    store.mark_downloaded(["arxiv:001", "arxiv:002"])
    assert store.downloaded_count == 2
    assert store.pending_count == 1
    assert store.get_pending_batch(10) == ["arxiv:003"]


def test_csv_column_is_doc_id(tmp_path):
    from backend.crawler.id_store import IdStore
    csv_path = tmp_path / "ids.csv"
    store = IdStore(csv_path)
    store.add_ids(["arxiv:001"])
    with csv_path.open() as f:
        header = f.readline().strip().split(",")
    assert header[0] == "doc_id"
    assert "arxiv_id" not in header


def test_persistence_across_reload(tmp_path):
    from backend.crawler.id_store import IdStore
    csv_path = tmp_path / "ids.csv"
    s1 = IdStore(csv_path)
    s1.add_ids(["arxiv:001", "arxiv:002", "arxiv:003"])
    s1.mark_downloaded(["arxiv:001"])
    s2 = IdStore(csv_path)
    assert s2.total == 3
    assert s2.downloaded_count == 1
    assert s2.pending_count == 2


def test_load_legacy_csv_with_arxiv_id_column(tmp_path):
    from backend.crawler.id_store import IdStore
    csv_path = tmp_path / "legacy.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arxiv_id", "discovered_at", "downloaded"])
        w.writeheader()
        w.writerow({"arxiv_id": "arxiv:legacy01", "discovered_at": "2024", "downloaded": "False"})
        w.writerow({"arxiv_id": "arxiv:legacy02", "discovered_at": "2024", "downloaded": "True"})
    store = IdStore(csv_path)
    assert store.total == 2
    assert store.downloaded_count == 1
    assert "arxiv:legacy01" in store.get_pending_batch(10)


def test_multisource_ids_coexist(tmp_path):
    from backend.crawler.id_store import IdStore
    store = IdStore(tmp_path / "ids.csv")
    store.add_ids(["arxiv:111", "fake:aaa", "semantic:xyz"])
    assert store.total == 3
    store.mark_downloaded(["arxiv:111"])
    pending = store.get_pending_batch(10)
    assert set(pending) == {"fake:aaa", "semantic:xyz"}


def test_thread_safety(tmp_path):
    from backend.crawler.id_store import IdStore
    store = IdStore(tmp_path / "ids.csv")
    errors = []

    def add_batch(prefix, n):
        try:
            store.add_ids([f"{prefix}:{i:04d}" for i in range(n)])
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=add_batch, args=(f"src{t}", 20)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.total == 100
