import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from exp1.scorers import collaborative_embeddings, fit_scorer, _torch_model


def fixture():
    return {'items': [{'id': item} for item in 'abcd'],
            'train_sequences': [list('abca'), list('bac'), list('adc')],
            'validation': [{'user': 'u1', 'history': list('abca'), 'positive': 'd'}],
            'test': [{'user': 'u1', 'history': list('abcad'), 'positive': 'b'}]}


class ScorerTest(unittest.TestCase):
    def test_train_only_popularity_and_catalog_order(self):
        data = fixture()
        scorer = fit_scorer('most_popular', data, {}, 7, Path('.'))
        np.testing.assert_allclose(np.exp(scorer.scores([])), [4, 2, 3, 1])
        data['validation'][0]['positive'] = 'a'
        data['test'] = [{'history': ['d'] * 100, 'positive': 'd'}]
        np.testing.assert_equal(fit_scorer('most_popular', data, {}, 7, Path('.')).scores([]),
                                scorer.scores([]))

    def test_knn_cosine_neighbor_cap_and_unknown_fallback(self):
        data = {'items': [{'id': i} for i in 'abcd'],
                'train_sequences': [list('ab'), list('ab'), list('ac')]}
        scorer = fit_scorer('item_knn', data, {'knn_neighbors': 1}, 7, Path('.'))
        self.assertEqual(scorer.similarity[0].nnz, 1)
        self.assertAlmostEqual(scorer.similarity[0, 1], np.sqrt(2 / 3))
        self.assertEqual(int(np.argmax(scorer.scores(['a']))), 1)
        np.testing.assert_equal(scorer.scores(['unknown']), scorer.popularity_logits)

    def test_empty_train_nonsequential_scorers_are_finite(self):
        data = {'items': [{'id': item} for item in 'ab'], 'train_sequences': []}
        for name in ('most_popular', 'item_knn'):
            scorer = fit_scorer(name, data, {}, 1, Path('.'))
            for history in ([], ['a']):
                self.assertTrue(np.isfinite(scorer.scores(history)).all())
            np.testing.assert_equal(scorer.item_embeddings(), np.zeros((2, 64)))

    def test_collaborative_vectors_ignore_validation_and_metadata(self):
        data = fixture()
        vectors = collaborative_embeddings(data, 6, 7)
        self.assertEqual(vectors.shape, (4, 6))
        data['test'], data['validation'] = [], []
        for item in data['items']:
            item['description'] = 'a changed item description'
        np.testing.assert_allclose(vectors, collaborative_embeddings(data, 6, 7))
        self.assertTrue(np.isfinite(vectors).all())

    def test_causal_encoder_cannot_see_future_and_padding_is_finite(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = _torch_model(4, {'d_model': 8, 'heads': 2, 'layers': 1,
                                 'dropout': 0., 'max_history': 5}).eval()
        with torch.no_grad():
            first = model.encode(torch.tensor([[1, 2, 3]]))
            second = model.encode(torch.tensor([[1, 4, 4]]))
            np.testing.assert_allclose(first[:, 0].numpy(), second[:, 0].numpy(), atol=1e-6)
            batch = model(torch.tensor([[1, 2, 0], [2, 3, 4]]))
            alone = model(torch.tensor([[1, 2]]))
            self.assertTrue(torch.isfinite(batch).all())
            np.testing.assert_allclose(batch[0].numpy(), alone[0].numpy(), atol=1e-6)

    def test_sasrec_training_reproducibility_and_test_noninterference(self):
        config = {'epochs': 2, 'batch_size': 3, 'd_model': 8, 'heads': 2,
                  'layers': 1, 'max_history': 4, 'dropout': 0., 'device': 'cpu'}
        with tempfile.TemporaryDirectory() as directory:
            data = fixture()
            scorer = fit_scorer('sasrec', data, config, 17, Path(directory))
            logits = scorer.scores(list('ab'))
            self.assertEqual(logits.shape, (4,))
            self.assertTrue(np.isfinite(logits).all())
            self.assertEqual(scorer.item_embeddings().shape, (4, 8))
            self.assertTrue((Path(directory) / 'sasrec_best.pt').exists())
            self.assertEqual(scorer.metadata['selection'], 'validation_ce')
            self.assertEqual(scorer.metadata['train_samples'], 7)
            self.assertEqual(scorer.metadata['best_ce'], min(
                row['validation_ce'] for row in scorer.metadata['loss_history']))
            changed = copy.deepcopy(data)
            changed['test'] = [{'history': ['d'] * 100, 'positive': 'd'}]
            again = fit_scorer('sasrec', changed, config, 17, Path(directory))
            np.testing.assert_array_equal(logits, again.scores(list('ab')))
            # History order actually changes the sequential representation.
            self.assertFalse(np.allclose(logits, scorer.scores(list('ba'))))
            np.testing.assert_equal(scorer.scores([]), scorer.popularity_logits)

    def test_empty_validation_and_no_train_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture()
            data['validation'] = [{'history': ['a'], 'positive': 'not-in-catalog'}]
            scorer = fit_scorer('sasrec', data,
                {'epochs': 1, 'device': 'cpu', 'd_model': 8, 'layers': 1},
                7, Path(directory))
            self.assertEqual(scorer.metadata['validation_samples'], 0)
            self.assertEqual(scorer.metadata['selection'], 'train_ce_fallback')
            data['train_sequences'] = [['a']]
            with self.assertRaisesRegex(ValueError, 'at least two'):
                fit_scorer('sasrec', data, {}, 7, Path(directory))

    def test_validation_only_selects_checkpoint_and_early_stopping(self):
        config = {'epochs': 1, 'device': 'cpu', 'd_model': 8, 'layers': 1,
                  'dropout': 0., 'max_history': 4}
        with tempfile.TemporaryDirectory() as directory:
            data = fixture()
            first = fit_scorer('sasrec', data, config, 11, Path(directory))
            changed = copy.deepcopy(data)
            changed['validation'][0]['positive'] = 'b'
            second = fit_scorer('sasrec', changed, config, 11, Path(directory))
            np.testing.assert_array_equal(first.scores(['a']), second.scores(['a']))
            self.assertNotEqual(first.metadata['best_ce'], second.metadata['best_ce'])
            # Compare to identically initialized untrained weights to verify updates.
            torch.manual_seed(11)
            initial = _torch_model(4, config)
            self.assertFalse(np.array_equal(first.item_embeddings(),
                                           initial.items.weight[1:].detach().numpy()))
            stopped = fit_scorer('sasrec', data,
                dict(config, epochs=10, lr=0., patience=1), 11, Path(directory))
            self.assertEqual(stopped.metadata['epochs_run'], 2)
            self.assertEqual(stopped.metadata['best_epoch'], 1)


if __name__ == '__main__':
    unittest.main()
