"""Tests del LSIModel — build, save/load, varianza explicada."""
from __future__ import annotations
import pytest


def test_lsi_model_build_stats(lsi_model):
    model, model_path, db_path = lsi_model
    assert len(model.doc_ids) == 5
    assert len(model.term_ids) > 0
    assert model.k == 3
    assert model.docs_latent.shape == (5, 3)
    assert len(model.df_map) > 0


def test_lsi_model_variance_explained(lsi_model):
    model, _, db_path = lsi_model
    from backend.retrieval.lsi_model import LSIModel
    # min_df=1 required: test corpus is too small for the default min_df=20
    stats = LSIModel(k=3).build(db_path=db_path, min_df=1)
    assert 0 < stats["var_explained"] <= 1.0


def test_lsi_model_save_and_load(lsi_model, tmp_path):
    model, model_path, _ = lsi_model
    from backend.retrieval.lsi_model import LSIModel
    model2 = LSIModel()
    model2.load(path=model_path)
    assert model2.k == 3
    assert len(model2.doc_ids) == 5
    assert model2.doc_ids == model.doc_ids
    assert model2.term_ids == model.term_ids
    assert model2.df_map == model.df_map
    assert model2.docs_latent.shape == (5, 3)