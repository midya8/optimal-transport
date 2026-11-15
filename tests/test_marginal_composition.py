"""Full-history finite-direction checks, not optimum-sensitivity guarantees."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

from transportlab.marginal import audit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('marginal_controls', ROOT / 'examples/marginal_controls.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)
TANGENTS = ('plan_tangent', 'log_plan_tangent', 'actual_source_tangent',
            'actual_target_tangent', 'source_potential_tangent', 'target_potential_tangent',
            'transport_tangent', 'full_tangent', 'direct_terms', 'requested_marginal_terms',
            'actual_mass_terms', 'envelope_candidate', 'residual_tangent_terms',
            'chain_identity_signed_residual', 'envelope_identity_signed_residual')


def combine(a, b, scale):
    if a is None:
        assert b is None
        return None
    if isinstance(a, list):
        return [combine(x, y, scale) for x, y in zip(a, b)]
    return a + scale * b


class MarginalCompositionTests(unittest.TestCase):
    def close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.close(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for x, y in zip(actual, expected):
                self.close(x, y)
        elif type(expected) is float:
            self.assertTrue(math.isclose(actual, expected, rel_tol=2e-8, abs_tol=2e-9),
                            (actual, expected))
        else:
            self.assertEqual(actual, expected)

    def test_direction_superposition_at_every_history_step(self):
        checked = 0
        for control in controls.controls():
            first = copy.deepcopy(control['input'])
            # Keep directions small even for the large formal underflow control.
            for key in ('source_direction', 'target_direction'):
                first[key] = [v / 8 for v in first[key]]
            first['cost_direction'] = [[v / 8 for v in row] for row in first['cost_direction']]
            first['epsilon_direction'] /= 8
            second = copy.deepcopy(first)
            for side in ('source', 'target'):
                second[side + '_direction'] = [0.125 * (-1)**i if w else 0.0
                                               for i, w in enumerate(second[side + '_weights'])]
            second['cost_direction'] = [[(i - j + 1) / 8 for j in range(len(row))]
                                         for i, row in enumerate(second['cost'])]
            second['epsilon_direction'] = -0.125
            a, b = audit(first), audit(second)
            for scale in (-1.0, 0.5, 2.0):
                mixed = copy.deepcopy(first)
                for key in ('source_direction', 'target_direction', 'cost_direction', 'epsilon_direction'):
                    mixed[key] = combine(first[key], second[key], scale)
                result = audit(mixed)
                with self.subTest(control=control['name'], scale=scale):
                    for key in ('source_tangent', 'target_tangent', 'raw_total_tangents'):
                        self.close(result[key], combine(a[key], b[key], scale))
                    for row, ar, br in zip(result['history'], a['history'], b['history']):
                        for key in TANGENTS:
                            self.close(row[key], combine(ar[key], br[key], scale))
                        # Absolute/max tangent residuals are not linear functions.
                        for key in row.keys() - set(TANGENTS) - {'row_tangent_residual', 'column_tangent_residual'}:
                            self.close(row[key], ar[key])
                checked += 1
        self.assertEqual(checked, 36)

    def test_adding_radial_raw_weight_directions_is_null(self):
        checked = 0
        for control in controls.controls():
            original = copy.deepcopy(control['input'])
            expected = audit(original)
            for radial in (-0.25, 0.25):
                changed = copy.deepcopy(original)
                for side in ('source', 'target'):
                    changed[side + '_direction'] = [d + radial * w for d, w in
                        zip(changed[side + '_direction'], changed[side + '_weights'])]
                result = audit(changed)
                with self.subTest(control=control['name'], radial=radial):
                    for key in expected.keys() - {'input', 'raw_total_tangents'}:
                        self.close(result[key], expected[key])
                    self.close(result['raw_total_tangents'],
                               [d + radial * total for d, total in
                                zip(expected['raw_total_tangents'], expected['raw_totals'])])
                checked += 1
        self.assertEqual(checked, 24)

    def test_raw_weight_and_direction_rescaling_preserves_full_history(self):
        checked = 0
        for control in controls.controls():
            original = copy.deepcopy(control['input'])
            expected = audit(original)
            for source_scale, target_scale in ((0.5, 2.0), (2.0, 0.5)):
                changed = copy.deepcopy(original)
                for side, scale in (('source', source_scale), ('target', target_scale)):
                    for suffix in ('_weights', '_direction'):
                        changed[side + suffix] = [scale * v for v in changed[side + suffix]]
                result = audit(changed)
                with self.subTest(control=control['name'], scales=(source_scale, target_scale)):
                    for key in expected.keys() - {'input', 'raw_totals', 'raw_total_tangents'}:
                        self.close(result[key], expected[key])
                    self.close(result['raw_totals'], [source_scale * expected['raw_totals'][0],
                                                     target_scale * expected['raw_totals'][1]])
                    self.close(result['raw_total_tangents'],
                               [source_scale * expected['raw_total_tangents'][0],
                                target_scale * expected['raw_total_tangents'][1]])
                checked += 1
        self.assertEqual(checked, 24)


if __name__ == '__main__':
    unittest.main()
