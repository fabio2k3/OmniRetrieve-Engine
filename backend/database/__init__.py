from .schema import DATA_DIR, DB_PATH, get_connection, init_db
from .crawler_repository import (
    upsert_document, save_pdf_text, save_pdf_error,
    get_pending_pdf_ids, get_document, document_exists,
    log_crawl_start, log_crawl_end, get_stats,
    get_document_counts, get_unindexed_pdf_count,
)
from .chunk_repository import (
    save_chunks, get_chunks,
    save_chunk_embedding, save_chunk_embeddings_batch,
    get_unembedded_chunks, get_unembedded_chunks_iter,
    get_all_embeddings_iter,
    get_chunk_count, get_embedded_count, get_chunk_stats,
    get_chunks_by_ids,
    reset_embeddings, get_chunks_with_metadata_by_arxiv_ids,
    get_chunk_embeddings_by_ids, 
)
from .embedding_repository import (
    init_embedding_schema,
    log_faiss_build,
    save_embedding_meta,
    get_embedding_meta,
    get_embedding_stats,
)
from .index_repository import (
    clear_index, upsert_terms, flush_postings, save_index_meta,
    get_unindexed_documents, mark_documents_indexed, get_index_stats,
    get_top_terms, get_postings_for_term,
    get_postings_for_matrix, get_document_metadata,
    get_indexed_doc_count, get_term_words_by_ids,
)

__all__ = [
    # schema
    "DATA_DIR", "DB_PATH", "get_connection", "init_db",
    # crawler_repository
    "upsert_document", "save_pdf_text", "save_pdf_error",
    "get_pending_pdf_ids", "get_document", "document_exists",
    "log_crawl_start", "log_crawl_end", "get_stats", "get_document_counts",
    "get_unindexed_pdf_count",
    # chunk_repository
    "save_chunks", "get_chunks",
    "save_chunk_embedding", "save_chunk_embeddings_batch",
    "get_unembedded_chunks", "get_unembedded_chunks_iter",
    "get_all_embeddings_iter",
    "get_chunk_count", "get_embedded_count", "get_chunk_stats",
    "get_chunks_by_ids",
    "reset_embeddings", "get_chunks_with_metadata_by_arxiv_ids",
    # embedding_repository
    "init_embedding_schema", "log_faiss_build",
    "save_embedding_meta", "get_embedding_meta", "get_embedding_stats",
    # index_repository
    "clear_index", "upsert_terms", "flush_postings", "save_index_meta",
    "get_unindexed_documents", "mark_documents_indexed", "get_index_stats",
    "get_top_terms", "get_postings_for_term",
    "get_postings_for_matrix", "get_document_metadata", ""
    "get_indexed_doc_count", "get_term_words_by_ids",
]