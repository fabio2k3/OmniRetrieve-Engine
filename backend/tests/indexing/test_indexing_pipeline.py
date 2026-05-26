"""Tests del IndexingPipeline — indexación inicial, incremental y reindex."""
from __future__ import annotations
import pytest


def test_pipeline_indexes_docs_with_pdf(pipeline):
    stats = pipeline.run(reindex=False)
    assert stats["docs_processed"] >= 3
    assert stats["terms_added"] > 0
    assert stats["postings_added"] > 0


def test_pipeline_skips_docs_without_pdf(pipeline):
    """El doc con pdf_downloaded=0 no debe indexarse."""
    stats = pipeline.run(reindex=False)
    # Solo 3 de los 4 docs tienen pdf_downloaded=1
    assert stats["docs_processed"] == 3


def test_incremental_does_not_reprocess(pipeline):
    pipeline.run(reindex=False)
    stats2 = pipeline.run(reindex=False)
    assert stats2["docs_processed"] == 0


def test_reindex_reprocesses_all(pipeline):
    pipeline.run(reindex=False)
    stats3 = pipeline.run(reindex=True)
    assert stats3["docs_processed"] >= 3
