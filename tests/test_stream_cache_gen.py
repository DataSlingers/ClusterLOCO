"""Regression checks against the original LOCO implementation.

Run with: python -m unittest discover -s tests -p 'test_stream_cache_gen.py'
"""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from clim.minipatches import ClusterLOCOMP, ClusterLOCOMPStream
from clim.minipatches.stream_cache_gen import _PredictionCache
from clim.utils import hinge_error


def label_errors(z, pred):
    return (z != pred).astype(float)


def scalar_error(z, pred):
    return float(np.mean(z != pred))


class StreamCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, _ = make_blobs(n_samples=72, n_features=30, centers=3, random_state=12)
        cls.original = cls.new_model(ClusterLOCOMP)
        with threadpool_limits(limits=1):
            cls.original.fit(cls.X, patch_n=24, patch_m=6, standardize=True,
                             parallel_MP=False, pprint=False)
        cls.cached = copy.copy(cls.original)
        cls.cached.__class__ = ClusterLOCOMPStream

    @staticmethod
    def new_model(cls=ClusterLOCOMPStream):
        return cls(K=3, B=21, random_state=17,
                   base_clusterer=KMeans(n_clusters=3, n_init=1),
                   base_classifier=DecisionTreeClassifier(max_depth=2))

    def assert_scores_equal(self, expected, actual):
        self.assertEqual(expected.keys(), actual.keys())
        for name, value in expected.items():
            if isinstance(value, np.ndarray) or isinstance(value, (float, int)):
                np.testing.assert_allclose(value, actual[name], rtol=1e-6,
                                           atol=1e-7, equal_nan=True, err_msg=name)
            else:
                self.assertEqual(value, actual[name], name)

    def test_all_score_modes_match(self):
        for metric, proba in [(hinge_error, True), (None, False),
                              (label_errors, False), (scalar_error, False)]:
            for agg in ["mean", "none", "by_clusters"]:
                for cache in ["memory", "disk"]:
                    with self.subTest(metric=metric, agg=agg, cache=cache):
                        kw = dict(error_metric=metric, proba_error=proba, agg=agg,
                                  features=[29, 0, 7, 0])
                        with contextlib.redirect_stderr(io.StringIO()):
                            expected = self.original.score(**kw)
                        self.assert_scores_equal(expected, self.cached.score(cache=cache, **kw))

    def test_predict_once_per_patch_and_refresh_each_call(self):
        with patch.object(self.cached, "_mp_predict_omp", wraps=self.cached._mp_predict_omp) as predict:
            self.cached.score(hinge_error)
            self.assertEqual(predict.call_count, self.cached.B)
            changed = self.X.copy()
            changed[:, 0] *= -1
            actual = self.cached.score(hinge_error, X=changed, cache="disk")
            self.assertEqual(predict.call_count, 2 * self.cached.B)
        with contextlib.redirect_stderr(io.StringIO()):
            expected = self.original.score(hinge_error, X=changed)
        self.assert_scores_equal(expected, actual)

    def test_stream_fit_matches_original_and_parallel(self):
        for jobs in [1, 2]:
            with self.subTest(n_jobs=jobs):
                model = self.new_model().fit_stream(
                    self.X, patch_n=24, patch_m=6, standardize=True,
                    n_jobs=jobs, patch_batch_size=4)
                for name in ["patch_rows_", "patch_cols_", "mp_labels_raw_", "mp_labels_", "z_ref"]:
                    np.testing.assert_array_equal(getattr(model, name), getattr(self.original, name))
                self.assert_scores_equal(self.cached.score(hinge_error), model.score(hinge_error))

    def test_inherited_fit_is_supported(self):
        with threadpool_limits(limits=1):
            model = self.new_model().fit(self.X, patch_n=24, patch_m=6,
                                        standardize=True, parallel_MP=False, pprint=False)
        self.assert_scores_equal(self.cached.score(hinge_error), model.score(hinge_error))

    def test_prediction_stream_is_lazy(self):
        with patch.object(self.cached, "_mp_predict_omp", wraps=self.cached._mp_predict_omp) as predict:
            stream = self.cached.iter_patch_predictions(patches=[3, 1])
            self.assertEqual(predict.call_count, 0)
            b, rows, values = next(stream)
            self.assertEqual(b, 3)
            self.assertEqual(predict.call_count, 1)
            expected_rows, expected_values = self.original._mp_predict_omp(self.original.X_fit_, 3, True)
            np.testing.assert_array_equal(rows, expected_rows)
            np.testing.assert_array_equal(values, expected_values)
            self.assertEqual(next(stream)[0], 1)
            with self.assertRaises(StopIteration):
                next(stream)

    def test_streamed_predictions_match(self):
        for proba in [True, False]:
            expected, _, _ = self.original._mp_loo(self.original.X_fit_, proba=proba)
            actual = self.cached.predict_proba() if proba else self.cached.predict()
            np.testing.assert_allclose(expected, actual, equal_nan=True)

    def test_cache_budget_and_error_cleanup(self):
        import clim.minipatches.stream_cache_gen as module
        opened = []
        real_temporary_file = tempfile.TemporaryFile

        def track_file(*args, **kwargs):
            f = real_temporary_file(*args, **kwargs)
            opened.append(f)
            return f

        def broken_metric(z, pred):
            raise RuntimeError("metric failed")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "TemporaryFile", side_effect=track_file):
                self.cached.score(hinge_error, max_cache_bytes=0, cache_dir=directory)
                with self.assertRaisesRegex(RuntimeError, "metric failed"):
                    self.cached.score(broken_metric, cache="disk", cache_dir=directory)
                with patch.object(self.cached, "_mp_predict_omp", side_effect=RuntimeError("prediction failed")):
                    with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                        self.cached.score(hinge_error, cache="disk", cache_dir=directory)
            self.assertEqual(len(opened), 3)
            self.assertTrue(all(f.closed for f in opened))
            self.assertEqual(list(Path(directory).iterdir()), [])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.cached.score(hinge_error, cache="memory", max_cache_bytes=0)

    def test_budget_boundary(self):
        size = self.cached.B * (72 - 24) * (3 + 1) * 4
        for budget, mode in [(size, "memory"), (size - 1, "disk")]:
            with _PredictionCache(21, 48, 3, True, "auto", budget, None) as cache:
                self.assertEqual(cache.mode, mode)
                self.assertEqual(cache.nbytes, size)

    def test_uncovered_rows_and_missing_classifier_classes(self):
        model = self.new_model()
        model.B = 2
        model.base_clusterer.set_params(n_clusters=1)
        model.fit_stream(self.X, patch_n=60, patch_m=6)
        self.assertTrue(all(len(f.classes_) < model.K for f in model.models_))
        original = copy.copy(model)
        original.__class__ = ClusterLOCOMP
        for metric in [hinge_error, None]:
            with contextlib.redirect_stderr(io.StringIO()):
                expected = original.score(metric)
            actual = model.score(metric, cache="disk")
            self.assert_scores_equal(expected, actual)
            self.assertTrue((actual["loo_count"] == 0).any())

    def test_all_rows_in_patch(self):
        model = self.new_model().fit_stream(self.X, patch_n=72, patch_m=6)
        for cache in ["memory", "disk"]:
            result = model.score(hinge_error, cache=cache)
            self.assertTrue(np.isnan(result["delta"]).all())
            self.assertTrue((result["loo_count"] == 0).all())

    def test_invalid_inputs(self):
        with self.assertRaises(NotFittedError):
            self.new_model().score(hinge_error)
        with self.assertRaises(NotFittedError):
            self.new_model().predict()
        for features in [[-1], [30], [1.5], [[1]]]:
            with self.assertRaises(ValueError):
                self.cached.score(features=features)
        for kw in [dict(X=self.X[:-1]), dict(parallel_features=True),
                   dict(cache="bad"), dict(max_cache_bytes=-1), dict(z=[1])]:
            with self.assertRaises(ValueError):
                self.cached.score(**kw)
        result = self.cached.score(hinge_error, features=[])
        self.assertEqual(result["delta"].shape, (0,))

    def test_many_features(self):
        X = np.random.default_rng(12).normal(size=(60, 2000))
        model = self.new_model()
        model.B = 12
        model.fit_stream(X, patch_n=20, patch_m=40)
        with patch.object(model, "_mp_predict_omp", wraps=model._mp_predict_omp) as predict:
            actual = model.score(hinge_error, cache="disk")
            self.assertEqual(predict.call_count, model.B)
        self.assertEqual(actual["delta"].shape, (2000,))
        original = copy.copy(model)
        original.__class__ = ClusterLOCOMP
        with contextlib.redirect_stderr(io.StringIO()):
            expected = original.score(hinge_error)
        self.assert_scores_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
