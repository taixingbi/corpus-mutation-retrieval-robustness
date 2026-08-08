from pathlib import Path

import numpy as np

from facm.retrieval.retrievers import _save_embed_cache, _try_load_embed_cache


def test_embed_cache_roundtrip(tmp_path: Path):
    ids = ["c1", "c2", "c3"]
    emb = np.random.randn(3, 8).astype(np.float32)
    _save_embed_cache(tmp_path, "org/model", "passage: ", ids, emb)
    loaded = _try_load_embed_cache(tmp_path, "org/model", "passage: ", ids)
    assert loaded is not None
    assert loaded.shape == (3, 8)
    np.testing.assert_allclose(loaded, emb)

    # different chunk ids → miss
    assert _try_load_embed_cache(tmp_path, "org/model", "passage: ", ["c1", "c2"]) is None
    # different doc prefix → miss
    assert _try_load_embed_cache(tmp_path, "org/model", "", ids) is None


def test_embed_cache_disabled():
    assert _try_load_embed_cache(None, "m", "", ["a"]) is None
