"""Figure 2 integration checks; optional legacy benchmarks are stubbed."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from clim.minipatches import ClusterLOCOMP, ClusterLOCOMPStream
from clim.utils import hinge_error
from scripts import figure2_experiments as figure2


class Figure2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(n_samples=96, n_features=20, centers=3, random_state=10)
        cls.clusterer = KMeans(n_clusters=3, n_init=1, random_state=10)

    def test_cached_helper_matches_original(self):
        with tempfile.TemporaryDirectory() as directory:
            scores, model, fit_time, score_time = figure2.fit_cached_cloc(
                self.X, K=3, B=17, base_clusterer=self.clusterer,
                prediction_cache="disk", cache_dir=directory, patch_batch_size=3)
            self.assertIsInstance(model, ClusterLOCOMPStream)
            self.assertEqual(model.B, 17)
            original = copy.copy(model)
            original.__class__ = ClusterLOCOMP
            with contextlib.redirect_stderr(io.StringIO()):
                expected = original.score(hinge_error)
            np.testing.assert_allclose(scores["delta"], expected["delta"], equal_nan=True)
            self.assertGreater(fit_time, 0)
            self.assertGreater(score_time, 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_simulation_uses_cache_for_cloc_and_rampart(self):
        # Exercise both real LOCO paths without the unrelated PyTorch benchmarks.
        values = np.linspace(0, 1, self.X.shape[1])
        shap = Mock()
        shap.return_value.get_model_wide_importance.return_value = (values, self.y)
        benchmarks = (Mock(return_value=values), Mock(return_value=values), shap,
                      Mock(return_value=pd.DataFrame({"featureImp": values})))
        stability = Mock()
        stability.return_value.impacc.return_value = dict(feature_importance=values, labels=self.y)
        split = Mock(return_value=(values, values))
        with patch.object(figure2, "_load_benchmarks", return_value=benchmarks), \
             patch.object(figure2, "GlobalStability_MP", stability), \
             patch.object(figure2, "Cluster_LOCO_Split", split), \
             patch.object(figure2, "fit_cached_cloc", wraps=figure2.fit_cached_cloc) as fit, \
             contextlib.redirect_stdout(io.StringIO()):
            result = figure2.run_one_simulation(
                X_aug=self.X, y=self.y, K=3, informative_d=10, noise_d=10,
                base_clusterer=self.clusterer, B=17, B_ramp=12, B_shapley=23,
                inner_n_jobs=2, prediction_cache="disk", patch_batch_size=4,
                max_cache_bytes=0, standardize=False)
        calls = fit.call_args_list
        self.assertGreater(len(calls), 1)
        self.assertEqual(calls[0].kwargs["B"], 17)
        self.assertEqual(calls[-1].kwargs["B"], 12)
        for call in calls:
            self.assertEqual(call.kwargs["inner_n_jobs"], 2)
            self.assertEqual(call.kwargs["prediction_cache"], "disk")
            self.assertEqual(call.kwargs["max_cache_bytes"], 0)
            self.assertFalse(call.kwargs["standardize"])
        self.assertEqual(split.call_args.kwargs["n_jobs"], 2)
        self.assertEqual(shap.call_args.kwargs["n_jobs"], 2)
        self.assertEqual(shap.call_args.kwargs["M"], 23)
        for key in ("cloc", "rampart"):
            self.assertTrue(np.isfinite(result["times"][key]))
            self.assertTrue(np.isfinite(result["ari"][key]))
        phases = result["phase_times"]
        self.assertGreaterEqual(result["times"]["cloc"], phases["cloc_fit"] + phases["cloc_score"])

    def test_worker_forwards_budget(self):
        with patch.object(figure2, "run_chunk") as run:
            figure2.worker(2, {}, 1, 123, "/tmp", inner_n_jobs=3)
        self.assertEqual(run.call_args.kwargs["inner_n_jobs"], 3)

    def test_chunk_preserves_output_keys_and_records_settings(self):
        methods = ["pbfi", "lrp", "impacc", "split_cloc", "cloc", "rampart", "perm", "cshap"]
        result = {key: {m: 0.5 for m in methods} for key in ("times", "ari", "topk_recall")}
        result["phase_times"] = {"cloc_fit": 0.2, "cloc_score": 0.3}
        cfg = dict(K=3, sim_method="moon-donut", informative_d=10,
                   noise_plan=[dict(type="gaussian", d=10)], n_per_cluster=32,
                   alpha=2, d0=2, gaps=[0.2]*3, shape_probs={"moon": 0.5, "donut": 0.5},
                   oversample=10, B=17, B_ramp=12, prediction_cache="disk", max_cache_bytes=0)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(figure2, "generate_dataset_for_one_run", return_value=(self.X, self.y)) as generate, \
             patch.object(figure2, "run_one_simulation", return_value=result) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            figure2.run_chunk(cfg=cfg, cfg_id=3, n_sims=1, task_id=2,
                              global_seed=123, out_dir=directory, inner_n_jobs=2)
            with np.load(Path(directory) / "results_task00002.npz") as saved:
                for m in methods:
                    for prefix in ("time", "ari", "topk_hits"):
                        self.assertEqual(saved[f"{prefix}_{m}"].shape, (1,))
                self.assertEqual(saved["time_cloc_fit"][0], 0.2)
                metadata = json.loads(saved["runtime_config"].item())
                self.assertEqual(metadata["inner_n_jobs"], 2)
                self.assertEqual(metadata["B"], 17)
                self.assertEqual(metadata["prediction_cache"], "disk")
                self.assertEqual(metadata["cloc_implementation"], "ClusterLOCOMPStream")
        self.assertEqual(generate.call_args.kwargs["sim_seed"], 123 + 3_000_000 + 20_000)
        self.assertEqual(run.call_args.kwargs["inner_n_jobs"], 2)
        self.assertIsInstance(run.call_args.kwargs["base_clusterer"], figure2.BaseSpectralClustering)


if __name__ == "__main__":
    unittest.main()
