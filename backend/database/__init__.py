from .schema import DB_PATH, get_connection, init_db
from .crawler_repository import (
    upsert_document, save_pdf_text, save_pdf_error,
    get_pending_pdf_ids, get_document, document_exists,
    save_chunks, get_chunks,
    get_unembedded_chunks, mark_chunk_embedded,
    log_crawl_start, log_crawl_end, get_stats,
)
from .index_repository import (
    clear_index, upsert_terms, flush_postings, save_index_meta,
    get_unindexed_documents, mark_documents_indexed, get_index_stats,
    get_top_terms, get_postings_for_term,
    get_postings_for_matrix, get_document_metadata,
)

__all__ = [
    # schema
    "DB_PATH", "get_connection", "init_db",
    # crawler_repository
    "upsert_document", "save_pdf_text", "save_pdf_error",
    "get_pending_pdf_ids", "get_document", "document_exists",
    "save_chunks", "get_chunks",
    "get_unembedded_chunks", "mark_chunk_embedded",
    "log_crawl_start", "log_crawl_end", "get_stats",
    # index_repository
    "clear_index", "upsert_terms", "flush_postings", "save_index_meta",
    "get_unindexed_documents", "mark_documents_indexed", "get_index_stats",
    "get_top_terms", "get_postings_for_term",
    "get_postings_for_matrix", "get_document_metadata",
]