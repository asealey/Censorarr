"""Unit tests for the MLX worker's pure helpers. Run: .venv/bin/python -m unittest test_worker_mlx"""
import unittest

import worker_mlx as w


def _words(*tokens):
    return [{'start': float(i), 'end': i + 0.5, 'word': t} for i, t in enumerate(tokens)]


class RepetitionLoopTests(unittest.TestCase):
    def test_keeps_short_legitimate_repeats(self):
        words = _words('fuck', 'fuck,', 'Fuck!', 'fuck', 'off')
        kept, dropped = w._drop_repetition_loops(words, 8)
        self.assertEqual((len(kept), dropped), (5, 0))

    def test_truncates_runaway_loop(self):
        words = _words('Kohidi', *(['da'] * 220), 'end')
        kept, dropped = w._drop_repetition_loops(words, 8)
        self.assertEqual(dropped, 212)
        self.assertEqual([x['word'] for x in kept], ['Kohidi'] + ['da'] * 8 + ['end'])

    def test_disabled_with_zero_limit(self):
        words = _words(*(['da'] * 20))
        self.assertEqual(w._drop_repetition_loops(words, 0), (words, 0))


class ModelResolutionTests(unittest.TestCase):
    def test_censorarr_names_map_to_mlx_community(self):
        self.assertEqual(w.resolve_model('small.en'), 'mlx-community/whisper-small.en-mlx')
        self.assertEqual(w.resolve_model('large-v3'), 'mlx-community/whisper-large-v3-mlx')
        self.assertEqual(w.resolve_model('large-v3-turbo'), 'mlx-community/whisper-large-v3-turbo')

    def test_repo_ids_pass_through(self):
        self.assertEqual(w.resolve_model('someone/whisper-x'), 'someone/whisper-x')


if __name__ == '__main__':
    unittest.main()
