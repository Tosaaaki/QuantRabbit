import importlib


def test_normalize_result_dict_int():
    m = importlib.import_module('cloudrun.news_summarizer_runner')
    res = m._normalize_result({"summary": "abc", "sentiment": "2"})
    assert res["summary"] == "abc"
    assert isinstance(res["sentiment"], int)
    assert res["sentiment"] == 2


def test_normalize_result_str():
    m = importlib.import_module('cloudrun.news_summarizer_runner')
    res = m._normalize_result("hello")
    assert res == {"summary": "hello", "sentiment": 0}

