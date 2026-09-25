"""One training-free policy. Its only queue input is Replay.visible() data.

The public request calendar is a bounded pressure proxy, not a calibrated ETA.
All pending jobs are conservatively placed ahead of the target in age order.
No private scheduler priority, actual background runtimes or future arrivals
are inputs. The same policy and constants are used for every synthetic trace.
"""
from bisect import bisect_right
from functools import lru_cache
import math


def _reserve(starts, free, nodes, hours):
    candidate = None
    for i, available in enumerate(free):
        if available < nodes:
            candidate = None
            continue
        if candidate is None:
            candidate = starts[i]
        stop = candidate + hours
        if i + 1 == len(starts) or starts[i + 1] >= stop:
            break
    if candidate is None:
        raise ValueError('Invalid public capacity calendar')
    stop = candidate + hours
    for boundary in (candidate, stop):
        i = bisect_right(starts, boundary) - 1
        if starts[i] != boundary:
            starts.insert(i + 1, boundary)
            free.insert(i + 1, free[i])
    for i in range(bisect_right(starts, candidate) - 1, bisect_right(starts, stop) - 1):
        free[i] -= nodes
        if free[i] < 0:
            raise ValueError('Negative public reservation capacity')
    return candidate


def public_pressure(snapshot, actions, request_hours, cap_hours):
    """Use only resource counts, elapsed wait and requested remaining time.

    Treating walltime as an upper occupancy limit is conservative only for the
    currently visible jobs. Future arrivals and private priority can invalidate
    it. Capping prevents a long pessimistic calendar dominating the objective.
    """
    capacity = snapshot['capacity_nodes']
    free_now = snapshot['available_nodes']
    if not 0 <= free_now <= capacity or any(n > capacity or n <= 0 for n in actions):
        raise ValueError('Invalid public node counts')
    releases = {}
    for job in snapshot['running']:
        hours = job['requested_remaining_seconds'] / 3600
        if not math.isfinite(hours) or hours < 0:
            raise ValueError('Invalid public remaining walltime')
        releases[hours] = releases.get(hours, 0) + job['nodes']
    if free_now + sum(releases.values()) != capacity:
        raise ValueError('Visible running nodes do not match visible available capacity')
    starts, free = [0.0], [free_now + releases.pop(0.0, 0)]
    for hours, nodes in sorted(releases.items()):
        starts.append(hours)
        free.append(free[-1] + nodes)
    # No private scheduler priority is reproduced here. Older visible pending
    # jobs are considered ahead of every new target request.
    pending = sorted(snapshot['pending'], key=lambda r: (-r['elapsed_wait_seconds'], -r['nodes']))
    for job in pending:
        if not 0 < job['nodes'] <= capacity or job['requested_seconds'] <= 0:
            raise ValueError('Invalid public pending request')
        _reserve(starts, free, job['nodes'], job['requested_seconds'] / 3600)
    return {n: min(cap_hours, _reserve(starts.copy(), free.copy(), n, request_hours)) for n in actions}


class OpportunityPolicy:
    def __init__(self, workload, pressure_cap_hours=48):
        self.work = workload
        self.nodes = tuple(sorted(workload.profiles))
        self.pressure_cap = pressure_cap_hours
        self.time_reference = min(self.completion_cost(n, workload.total_updates, True)[0] for n in self.nodes)
        self.nodehour_reference = min(self.completion_cost(n, workload.total_updates, True)[1] for n in self.nodes)

    @lru_cache(maxsize=8192)
    def completion_cost(self, nodes, remaining, first):
        """Finish this same remaining work at n, including all chunk overheads.

        This describes target execution using the given scaling profile. It
        does not replay or look ahead into the background workload.
        """
        hours = 0.0
        while remaining:
            plan = self.work.plan(nodes, remaining, first)
            if plan is None:
                return math.inf, math.inf
            hours += plan.actual.total_seconds() / 3600
            remaining -= plan.updates
            first = False
        return hours, nodes * hours

    def select(self, snapshot, remaining, first, alpha):
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        pressure = public_pressure(snapshot, self.nodes, float(self.work.max_request) / 3600, self.pressure_cap)
        candidates = []
        for n in self.nodes:
            hours, nodehours = self.completion_cost(n, remaining, first)
            score = alpha * (hours + pressure[n]) / self.time_reference + (1 - alpha) * nodehours / self.nodehour_reference
            candidates.append({'nodes': n, 'remaining_execution_hours': hours,
                               'remaining_nodehours': nodehours, 'pressure_hours': pressure[n], 'score': score})
        best = min(candidates, key=lambda r: (r['score'], r['remaining_nodehours'], r['nodes']))
        return best['nodes'], {'alpha': alpha, 'remaining_updates': remaining,
                              'public_snapshot': snapshot, 'candidates': candidates}
