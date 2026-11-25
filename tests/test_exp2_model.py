import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from exp2.model import (BOS, EOS, PAD, Generator, build_training_samples,
                        fit_generator, load_generator, _atomic_write)


def fixture():
    return {'items': [{'id': item} for item in 'abcd'],
            'train_sequences': [list('abca'), list('bac'), list('adc')],
            'validation': [{'user': 'u', 'history': list('abc'), 'positive': 'd'}],
            'test': [{'user': 'u', 'history': list('abcd'), 'positive': 'a'}]}


CODES = {'a': (0,), 'b': (0, 9), 'c': (4, 9), 'd': (4, 9, 11)}
CONFIG = {'d_model': 8, 'heads': 2, 'encoder_layers': 1, 'decoder_layers': 1,
          'max_history': 3, 'epochs': 2, 'batch_size': 3, 'dropout': 0.,
          'device': 'cpu', 'torch_threads': 2, 'max_train_samples': 0}


class GeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.output = Path(cls.directory.name)
        cls.generator = fit_generator(fixture(), CODES, CONFIG, 17, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_train_samples_are_strictly_earlier_and_ignore_heldout(self):
        data = fixture()
        self.assertEqual(build_training_samples(data, 2), [
            (['a'], 'b'), (['a', 'b'], 'c'), (['b', 'c'], 'a'),
            (['b'], 'a'), (['b', 'a'], 'c'), (['a'], 'd'), (['a', 'd'], 'c')])
        before = build_training_samples(data)
        data['validation'] = data['test'] = [{'history': ['d'] * 100, 'positive': 'd'}]
        self.assertEqual(before, build_training_samples(data))

    def test_complete_vocab_variable_prefix_batches_and_eos(self):
        generator = self.generator
        self.assertEqual(generator.item_ids, list('abcd'))
        self.assertEqual(generator.raw_token_mapping, {0: 3, 4: 4, 9: 5, 11: 6})
        self.assertTrue(all(code[-1] == EOS for code in generator.token_codes.values()))
        prefixes = [(), generator.encode_item('a')[:-1], generator.encode_item('d')[:-1]]
        batch = generator.log_probs(list('ab'), prefixes)
        self.assertEqual(batch.shape, (3, 7))
        self.assertTrue(np.isfinite(batch).all())
        np.testing.assert_allclose(np.exp(batch).sum(axis=1), 1., atol=1e-6)
        for i, prefix in enumerate(prefixes):
            alone = generator.log_probs(list('ab'), [prefix])[0]
            np.testing.assert_allclose(batch[i], alone, atol=1e-6)
            self.assertAlmostEqual(batch[i, EOS], alone[EOS], places=6)
        for invalid in ((EOS,), (BOS,), (PAD,), (3.5,), (True,)):
            with self.assertRaises(ValueError):
                generator.log_probs(['a'], [invalid])

    def test_causal_decoder_and_cross_attention_history(self):
        model = self.generator.model.eval()
        with torch.no_grad():
            histories = torch.tensor([[1, 2]])
            first = model(histories, torch.tensor([[BOS, 3, 4]]))
            second = model(histories, torch.tensor([[BOS, 5, 6]]))
            np.testing.assert_allclose(first[:, 0].numpy(), second[:, 0].numpy(), atol=1e-6)
        self.assertFalse(np.allclose(self.generator.log_probs(['a'], [()]),
                                     self.generator.log_probs(['b'], [()])))

    def test_teacher_forcing_contains_eos_and_ignores_padding(self):
        generator = self.generator
        history, inputs, targets = generator._batch([([], 'a'), (['a', 'b'], 'd')])
        self.assertEqual(targets.tolist(), [[3, EOS, PAD, PAD], [4, 5, 6, EOS]])
        self.assertEqual(inputs.tolist(), [[BOS, 3, PAD, PAD], [BOS, 4, 5, 6]])
        with torch.no_grad():
            logits = generator.model(history, inputs)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                                      ignore_index=PAD, reduction='sum')
            mask = targets.ne(PAD)
            explicit = -torch.log_softmax(logits, -1)[mask].gather(1, targets[mask, None]).sum()
            self.assertAlmostEqual(float(loss), float(explicit), places=5)
        self.assertTrue(np.isfinite(generator.log_probs([], [(), (3,)])).all())

    def test_checkpoint_load_without_train_and_best_validation(self):
        data = fixture()
        data['train_sequences'] = []
        loaded = load_generator(data, CODES, {'device': 'cpu'}, 17, self.output)
        np.testing.assert_array_equal(self.generator.log_probs(['a'], [(), (3,)]),
                                      loaded.log_probs(['a'], [(), (3,)]))
        self.assertTrue(loaded.metadata['loaded_from_checkpoint'])
        metadata = self.generator.metadata
        self.assertEqual(metadata['best_ce'], min(row['validation_ce'] for row in metadata['loss_history']))
        changed_codes = dict(CODES, a=(100,))
        with self.assertRaisesRegex(ValueError, 'codes|mapping'):
            load_generator(data, changed_codes, {}, 17, self.output)

    def test_invalid_codes_and_numpy_integer_support(self):
        for invalid in ((), (-1,), (1.0,), (True,), ('1',), (None,), None, (np.bool_(True),)):
            with self.subTest(code=invalid), self.assertRaisesRegex(ValueError, 'nonnegative integers'):
                Generator(fixture(), dict(CODES, a=invalid), CONFIG, 17)
        accepted = Generator(fixture(), dict(CODES, a=(np.int64(0),)), CONFIG, 17)
        self.assertEqual(accepted.token_codes, self.generator.token_codes)

    def test_restore_rejects_seed_and_configuration_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'seed'):
            load_generator(fixture(), CODES, {}, 999, self.output)
        with self.assertRaisesRegex(ValueError, 'config differs for d_model'):
            load_generator(fixture(), CODES, {'d_model': 16}, 17, self.output)
        restored = load_generator(fixture(), CODES, dict(CONFIG, torch_threads=1), 17, self.output)
        self.assertEqual(restored.metadata['parameter_count'],
                         sum(parameter.numel() for parameter in restored.model.parameters()))
        self.assertGreaterEqual(restored.metadata['train_seconds'], 0.)
        self.assertIsNone(restored.metadata['peak_cuda_memory_bytes'])
        self.assertFalse(list(self.output.glob('*.tmp')))

    def test_atomic_artifact_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            path.write_bytes(b'previous complete checkpoint')
            def interrupted(temporary):
                temporary.write_bytes(b'partial replacement')
                raise RuntimeError('simulated interruption')
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                _atomic_write(path, interrupted)
            self.assertEqual(path.read_bytes(), b'previous complete checkpoint')
            self.assertEqual(list(Path(directory).iterdir()), [path])
            _atomic_write(path, lambda temporary: temporary.write_bytes(b'new complete checkpoint'))
            self.assertEqual(path.read_bytes(), b'new complete checkpoint')

    def test_reproducibility_training_updates_and_test_noninterference(self):
        changed = copy.deepcopy(fixture())
        changed['test'] = [{'history': ['d'] * 100, 'positive': 'd'}]
        with tempfile.TemporaryDirectory() as directory:
            again = fit_generator(changed, CODES, CONFIG, 17, Path(directory))
            np.testing.assert_array_equal(self.generator.log_probs(['a'], [()]),
                                          again.log_probs(['a'], [()]))
        initial = Generator(fixture(), CODES, CONFIG, 17)
        self.assertFalse(torch.equal(initial.model.output.weight, self.generator.model.output.weight))


if __name__ == '__main__':
    unittest.main()
