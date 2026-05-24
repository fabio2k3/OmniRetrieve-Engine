"""
test_factory.py
===============
Tests de factory.py — importaciones al nivel de módulo para que patch() funcione.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestBuildFaissManager:
    def test_raises_if_index_not_found(self):
        from backend.retrieval import factory
        import sys

        mock_mgr = MagicMock()
        mock_mgr.load.return_value = False
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value.get_sentence_embedding_dimension.return_value = 384

        with patch.object(factory, "FaissIndexManager", return_value=mock_mgr), \
             patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            with pytest.raises(RuntimeError, match="FAISS"):
                factory.build_faiss_manager(embed_model="mock-model")

    def test_returns_manager_when_loaded(self):
        from backend.retrieval import factory
        import sys

        mock_mgr = MagicMock()
        mock_mgr.load.return_value = True
        mock_mgr.total_vectors = 1000
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value.get_sentence_embedding_dimension.return_value = 384

        with patch.object(factory, "FaissIndexManager", return_value=mock_mgr), \
             patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
            mgr = factory.build_faiss_manager(embed_model="mock-model")

        assert mgr is mock_mgr


class TestBuildLsiRetriever:
    def test_raises_if_model_not_found(self, tmp_path):
        from backend.retrieval import factory

        with patch.object(factory, "MODEL_PATH", tmp_path / "nonexistent.pkl"):
            with pytest.raises(RuntimeError, match="LSI"):
                factory.build_lsi_retriever()

    def test_calls_load_on_retriever(self, tmp_path):
        from backend.retrieval import factory

        model_path = tmp_path / "lsi_model.pkl"
        model_path.touch()
        mock_retriever = MagicMock()

        with patch.object(factory, "MODEL_PATH", model_path), \
             patch.object(factory, "LSIRetriever", return_value=mock_retriever):
            factory.build_lsi_retriever()

        mock_retriever.load.assert_called_once()


class TestBuildEmbeddingRetriever:
    def test_reuses_provided_faiss_manager(self):
        from backend.retrieval import factory

        mock_faiss = MagicMock()
        mock_er = MagicMock()

        with patch.object(factory, "EmbeddingRetriever", return_value=mock_er) as MockER:
            result = factory.build_embedding_retriever(embed_model="mock-model", faiss_mgr=mock_faiss)

        MockER.assert_called_once_with(faiss_mgr=mock_faiss, model_name="mock-model")
        assert result is mock_er

    def test_builds_faiss_if_none_provided(self):
        from backend.retrieval import factory

        mock_faiss = MagicMock()
        mock_er = MagicMock()

        with patch.object(factory, "build_faiss_manager", return_value=mock_faiss), \
             patch.object(factory, "EmbeddingRetriever", return_value=mock_er):
            result = factory.build_embedding_retriever(embed_model="mock-model", faiss_mgr=None)

        assert result is mock_er


class TestBuildHybridRetriever:
    def _patches(self):
        from backend.retrieval import factory
        return (
            patch.object(factory, "build_faiss_manager",       return_value=MagicMock()),
            patch.object(factory, "build_lsi_retriever",       return_value=MagicMock()),
            patch.object(factory, "build_embedding_retriever", return_value=MagicMock()),
            patch.object(factory, "HybridRetriever",           return_value=MagicMock()),
            patch.object(factory, "CrossEncoderReranker",      return_value=MagicMock()),
        )

    def test_returns_hybrid_retriever(self):
        from backend.retrieval import factory
        p = self._patches()
        with p[0], p[1], p[2], p[3] as MockHybrid, p[4]:
            result = factory.build_hybrid_retriever(embed_model="mock-model")
        assert result is MockHybrid.return_value

    def test_reranker_passed_when_enabled(self):
        from backend.retrieval import factory
        p = self._patches()
        with p[0], p[1], p[2], p[3] as MockHybrid, p[4] as MockRerank:
            factory.build_hybrid_retriever(embed_model="mock-model", with_reranker=True)
        _, kwargs = MockHybrid.call_args
        assert kwargs.get("reranker") is MockRerank.return_value

    def test_no_reranker_when_disabled(self):
        from backend.retrieval import factory
        p = self._patches()
        with p[0], p[1], p[2], p[3] as MockHybrid, p[4]:
            factory.build_hybrid_retriever(embed_model="mock-model", with_reranker=False)
        _, kwargs = MockHybrid.call_args
        assert kwargs.get("reranker") is None

    def test_custom_rrf_k_forwarded(self):
        from backend.retrieval import factory
        p = self._patches()
        with p[0], p[1], p[2], p[3] as MockHybrid, p[4]:
            factory.build_hybrid_retriever(embed_model="mock-model", rrf_k=30)
        _, kwargs = MockHybrid.call_args
        assert kwargs.get("rrf_k") == 30

    def test_faiss_built_once_and_shared(self):
        from backend.retrieval import factory
        p = self._patches()
        with p[0] as mock_build_faiss, p[1], p[2] as mock_build_dense, p[3], p[4]:
            factory.build_hybrid_retriever(embed_model="mock-model")
        mock_build_faiss.assert_called_once()
        _, kwargs = mock_build_dense.call_args
        assert "faiss_mgr" in kwargs