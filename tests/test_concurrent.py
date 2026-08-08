from facm.concurrent import map_concurrent


def test_map_concurrent_preserves_order():
    out = map_concurrent(range(10), lambda x: x * 2, max_concurrency=4)
    assert out == [i * 2 for i in range(10)]


def test_map_concurrent_sequential():
    out = map_concurrent(range(5), lambda x: x + 1, max_concurrency=1)
    assert out == [1, 2, 3, 4, 5]
