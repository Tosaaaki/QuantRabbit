import importlib
import sys
import types
import os


class _StubBucket:
    def __init__(self, name):
        self.name = name


class _StubClient:
    def bucket(self, name):
        return _StubBucket(name)


def _install_storage_stub(monkeypatch):
    # Create stub module google.cloud.storage with Client
    storage = types.SimpleNamespace(Client=_StubClient)
    cloud = types.SimpleNamespace(storage=storage)
    google = types.SimpleNamespace(cloud=cloud)
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.cloud', cloud)
    monkeypatch.setitem(sys.modules, 'google.cloud.storage', storage)


def test_bucket_resolution_env_first(monkeypatch):
    _install_storage_stub(monkeypatch)
    monkeypatch.setenv('BUCKET', 'env-bucket')
    # ensure other envs not set
    monkeypatch.delenv('BUCKET_NEWS', raising=False)
    # reload module
    if 'market_data.news_fetcher' in sys.modules:
        del sys.modules['market_data.news_fetcher']
    mf = importlib.import_module('market_data.news_fetcher')
    assert mf.BUCKET == 'env-bucket'


def test_bucket_resolution_fallback_default(monkeypatch):
    _install_storage_stub(monkeypatch)
    monkeypatch.delenv('BUCKET', raising=False)
    monkeypatch.delenv('BUCKET_NEWS', raising=False)
    # reload module
    if 'market_data.news_fetcher' in sys.modules:
        del sys.modules['market_data.news_fetcher']
    mf = importlib.import_module('market_data.news_fetcher')
    assert mf.BUCKET == 'quantrabbit-fx-news'
