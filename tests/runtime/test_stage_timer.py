from omega_playground.policy.runner import StageTimer


def test_cuda_timer_defers_synchronization_until_report(monkeypatch):
    events, syncs = [], []

    class Event:
        def __init__(self, **kwargs):
            events.append(self)

        def record(self):
            pass

        def elapsed_time(self, other):
            return 2.5

    monkeypatch.setattr('torch.cuda.Event', Event)
    monkeypatch.setattr('torch.cuda.synchronize', lambda: syncs.append(True))
    timer = StageTimer(True)
    for _ in range(3):
        with timer('physics'):
            assert not syncs
    assert not syncs
    report = timer.report()['physics']
    assert len(syncs) == 1 and len(events) == 6
    assert report['count'] == 3 and report['mean_ms'] == 2.5
    assert report['measurement'] == 'cuda_events'
