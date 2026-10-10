"""Focused checks for sampled archive object summaries."""

from unittest import TestCase

from frigate.api.external_detection import collect_objects


class TestExternalDetection(TestCase):
    def test_tracked_labels_thresholds_and_best_frame(self):
        samples = [
            (
                0,
                [
                    ("person", 0.8, (0.1, 0.2, 0.9, 0.7)),
                    ("dog", 0.4, (0.1, 0.1, 0.3, 0.3)),
                    ("car", 0.99, (0, 0, 1, 1)),
                ],
            ),
            (
                2,
                [
                    ("person", 0.95, (0.2, 0.3, 0.8, 0.9)),
                    ("dog", 0.9, (0.1, 0.1, 0.3, 0.3)),
                ],
            ),
        ]
        objects = collect_objects(
            samples, ["person", "dog"], {"person": 0.7, "dog": 0.7}
        )
        self.assertEqual([obj["label"] for obj in objects], ["dog", "person"])
        person = objects[1]
        self.assertEqual(person["samples"], 2)
        self.assertEqual(person["first_offset"], 0)
        self.assertEqual(person["last_offset"], 2)
        self.assertEqual(person["snapshot_offset"], 2)
        self.assertEqual(person["score"], 0.95)
        self.assertAlmostEqual(person["box"][0], 0.3)
        self.assertAlmostEqual(person["box"][2], 0.6)

    def test_empty_inference_is_not_a_synthetic_doorbell_detection(self):
        self.assertEqual(
            collect_objects([(0, []), (1, [])], ["person"], {"person": 0.7}), []
        )
