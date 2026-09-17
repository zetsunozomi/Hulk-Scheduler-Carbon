"""A completion-budget slider with explicitly supported, validated ticks."""
import math
from .common import require


def slider_contract(time_reference_hours, budgets):
    grid = sorted(float(b) for b in budgets)
    require(grid and len(grid) == len(set(grid)) and all(math.isfinite(b) and b > 0 for b in grid),
            'Slider requires distinct positive budget multipliers')
    require(math.isfinite(time_reference_hours) and time_reference_hours > 0, 'Invalid slider time reference')
    width = grid[-1] - grid[0]
    return {'kind':'completion_budget_slider_v1', 'direction':'0 tighter completion budget; 1 more time allowed',
            'time_reference_hours':time_reference_hours,
            'ticks':[{'position':(b-grid[0])/width if width else 0., 'budget_multiplier':b,
                      'budget_hours':b*time_reference_hours} for b in grid],
            'between_ticks':'unsupported until trained and validated; no outcome interpolation',
            'claim':'budget target, not a guaranteed finish time or monotone learned curve'}


def budget_at_position(position, budgets):
    require(math.isfinite(position) and 0 <= position <= 1, 'Slider position must be in [0,1]')
    ticks = slider_contract(1., budgets)['ticks']
    matches = [t['budget_multiplier'] for t in ticks if math.isclose(position,t['position'],rel_tol=0,abs_tol=1e-9)]
    require(len(matches)==1, 'Unsupported slider position; use a declared tick, not curve interpolation')
    return matches[0]
