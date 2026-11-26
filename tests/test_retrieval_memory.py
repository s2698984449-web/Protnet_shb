from __future__ import annotations

import unittest

import torch

from training.retrieval_memory import (
    adaptive_topk,
    appearance_key,
    box_weighted_mean,
    clip_large_delta,
    difference_key,
    expected_members_per_bin,
    family_task_weights,
    fit_fixed_pca,
    geometry_descriptor,
    load_retrieval_config,
    median_anchor,
    merge_cluster_members,
    merge_groups,
    mix_cards,
    morphology_key,
    project_l2,
    retrieval_concentration,
    select_keep_ratio,
    spherical_kmeans,
    trim_vector,
)


class RetrievalMemoryTest(unittest.TestCase):
    def test_config_and_bin_expectation_follow_the_revised_defaults(self):
        config = load_retrieval_config()
        self.assertFalse(config["train_projection"])
        self.assertEqual(config["pca_dim"], 64)
        self.assertEqual(config["clusters_per_module"], 8)
        self.assertEqual(config["tasks_per_family"], 8)
        self.assertAlmostEqual(expected_members_per_bin(27, 8, 8, 3), 9.0)

    def test_signature_dimensions_and_reference_boundary(self):
        reference = torch.zeros(4, 256)
        reference[:, 0] = 2
        defect = torch.zeros(256)
        defect[1] = 4
        appearance = appearance_key(reference)
        self.assertEqual(tuple(appearance.shape), (256,))
        self.assertAlmostEqual(float(appearance[0]), 2.0)
        difference = difference_key(defect, appearance)
        self.assertAlmostEqual(float(difference[1]), 4.0)
        boxes = [torch.tensor([[0.0, 0.0, 10.0, 5.0], [0.0, 0.0, 2.0, 2.0]])]
        geometry = geometry_descriptor(boxes, (20, 20))
        self.assertEqual(tuple(geometry.shape), (6,))
        morphology = morphology_key(defect, geometry)
        self.assertEqual(tuple(morphology.shape), (262,))
        weighted = box_weighted_mean(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([3.0, 1.0]))
        self.assertAlmostEqual(float(weighted[0]), 0.75)

    def test_pca_is_shared_and_does_not_amplify_a_zero_vector(self):
        keys = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
        pca = fit_fixed_pca(keys, 2)
        projected = torch.stack([project_l2(row, pca) for row in keys])
        self.assertEqual(tuple(projected.shape), (4, 2))
        self.assertTrue(torch.allclose(projected.norm(dim=1), torch.ones(4), atol=1e-5))
        zero = project_l2(pca["mean"], pca)
        self.assertEqual(float(zero.norm()), 0.0)

    def test_kmeans_is_deterministic_and_marks_unused_clusters_invalid(self):
        left = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        right = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]])
        vectors = torch.cat([left, right])
        families = ["a", "a", "a", "b", "b", "b"]
        first = spherical_kmeans(vectors, families, 2, seed=7)
        second = spherical_kmeans(vectors, families, 2, seed=7)
        self.assertTrue(torch.equal(first["assignment"], second["assignment"]))
        self.assertEqual(len(set(int(item) for item in first["assignment"][:3])), 1)
        self.assertEqual(len(set(int(item) for item in first["assignment"][3:])), 1)
        self.assertNotEqual(int(first["assignment"][0]), int(first["assignment"][3]))
        unused = spherical_kmeans(vectors, families, 8, seed=7)
        self.assertEqual(unused["clusters_requested"], 8)
        self.assertGreaterEqual(int((~unused["valid"]).sum()), 2)
        weights = family_task_weights(["a", "a", "b", "b"])
        self.assertTrue(torch.allclose(weights, torch.full((4,), 0.5)))

    def test_clip_shrinks_only_large_deltas(self):
        anchor = torch.tensor(2.0)
        small, small_scale = clip_large_delta(torch.tensor([0.3, 0.4]), anchor)
        large, large_scale = clip_large_delta(torch.tensor([30.0, 40.0]), anchor)
        self.assertAlmostEqual(float(small_scale), 1.0)
        self.assertTrue(torch.allclose(small, torch.tensor([0.3, 0.4])))
        self.assertLess(float(large_scale), 1.0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(large)), 2.0, places=5)
        norms = [torch.tensor([1.0, 3.0, 5.0]), torch.tensor([10.0, 4.0, 8.0])]
        self.assertAlmostEqual(float(median_anchor(norms)), 5.5)

    def test_trim_keeps_the_largest_coordinates(self):
        delta = torch.tensor([0.1, -4.0, 0.2, 3.0])
        kept = trim_vector(delta, 0.5)
        self.assertEqual(int((kept != 0).sum()), 2)
        self.assertAlmostEqual(float(kept[1]), -4.0)
        self.assertAlmostEqual(float(kept[3]), 3.0)

    def test_ties_election_tie_and_small_sample_fallback(self):
        aligned = torch.tensor([[1.0, 1.0, 1.0]] + [[1.0, -1.0, 0.0]] * 4)
        merged = merge_cluster_members(aligned, torch.ones(5), keep_ratio=1.0, min_members=5)
        self.assertFalse(merged["low_sample"])
        self.assertAlmostEqual(float(merged["card"][0]), 1.0, places=5)
        self.assertAlmostEqual(float(merged["card"][1]), -1.0, places=5)
        self.assertAlmostEqual(float(merged["card"][2]), 1.0, places=5)
        tied = torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, 0.0]])
        election = merge_cluster_members(tied, torch.ones(5), keep_ratio=1.0, min_members=5)
        self.assertAlmostEqual(float(election["card"][1]), 0.0, places=5)
        fallback = merge_cluster_members(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), torch.ones(2), 1.0, min_members=5)
        self.assertTrue(fallback["low_sample"])
        self.assertAlmostEqual(float(fallback["unit"][0]), 1.0, places=5)
        self.assertAlmostEqual(float(fallback["unit"][1]), 0.0, places=5)

    def test_cross_group_merge_excludes_the_task_group_and_stays_unnormalized(self):
        units = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])]
        merged = merge_groups(units, [4, 4, 4], [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)], excluded_group=0)
        self.assertTrue(torch.allclose(merged["vector"], torch.tensor([0.0, 1.0])))
        mixed = merge_groups(
            [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
            [1, 1], [torch.tensor(1.0), torch.tensor(1.0)], excluded_group=None,
        )
        self.assertLess(float(torch.linalg.vector_norm(mixed["vector"])), 1.0)

    def test_topk_covers_zero_one_two_and_expands_when_mass_is_low(self):
        empty = adaptive_topk(torch.empty(0))
        self.assertEqual(empty["indices"], [])
        single = adaptive_topk(torch.tensor([0.2]))
        self.assertEqual(single["indices"], [0])
        self.assertAlmostEqual(single["concentration"], 1.0)
        pair = adaptive_topk(torch.tensor([0.2, 0.1]))
        self.assertEqual(len(pair["indices"]), 2)
        peaked = torch.tensor([1.0, 1.0, 0.0, 0.0])
        tight = adaptive_topk(peaked, mass_threshold=0.8, max_k=4)
        self.assertEqual(len(tight["indices"]), 2)
        flat = torch.linspace(-0.01, 0.01, 8)
        wide = adaptive_topk(flat, mass_threshold=0.8, max_k=4)
        self.assertEqual(len(wide["indices"]), 4)
        self.assertLess(wide["concentration"], 0.25)
        uniform = retrieval_concentration(torch.full((8,), 0.125))
        self.assertAlmostEqual(uniform, 0.0, places=5)

    def test_mix_cards_is_weighted_sum(self):
        mixed = mix_cards([torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])], torch.tensor([0.25, 0.75]))
        self.assertTrue(torch.allclose(mixed, torch.tensor([0.25, 0.75])))

    def test_keep_ratio_uses_split_half_and_falls_back_without_bins(self):
        stable = torch.ones(8, 4)
        unstable = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 4, dtype=torch.float32)
        chosen = select_keep_ratio({0.1: [unstable], 1.0: [stable]}, min_members=8)
        self.assertEqual(chosen["keep_ratio"], 1.0)
        self.assertTrue(chosen["calibrated"])
        fallback = select_keep_ratio({0.2: [torch.ones(3, 2)]}, min_members=8)
        self.assertEqual(fallback["keep_ratio"], 1.0)
        self.assertFalse(fallback["calibrated"])


if __name__ == "__main__":
    unittest.main()