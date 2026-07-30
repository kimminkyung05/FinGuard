"""Tests for the independent video decision layer."""

import json
import unittest

from video.service.aggregation import aggregate_frame_scores
from video.service.confidence import calculate_confidence
from video.service.output import build_output, dumps_output
from video.service.risk import assess_risk


class VideoServiceTest(unittest.TestCase):
    def _confidence(self, score, blur=0.1, face_frames=100):
        return calculate_confidence(100, 100, face_frames, [score] * 100, blur, 0.5)

    def test_low_fake_high_confidence_is_approved(self):
        aggregation = aggregate_frame_scores([0.1] * 100)
        confidence = self._confidence(0.1)
        result = assess_risk(aggregation["video_fake_score"], confidence["confidence_score"])
        self.assertEqual("APPROVE", result["decision"])

    def test_high_fake_high_confidence_is_blocked(self):
        aggregation = aggregate_frame_scores([0.9] * 100)
        confidence = self._confidence(0.9)
        result = assess_risk(aggregation["video_fake_score"], confidence["confidence_score"])
        self.assertEqual("BLOCK", result["decision"])

    def test_low_fake_low_confidence_retries(self):
        confidence = calculate_confidence(2, 1, 0, [], 0.9, 0.0)
        result = assess_risk(0.1, confidence["confidence_score"])
        self.assertEqual("RETRY", result["decision"])

    def test_empty_frames_retries(self):
        aggregation = aggregate_frame_scores([])
        confidence = calculate_confidence(0, 0, 0, [], 0.0, 0.5)
        result = assess_risk(aggregation["video_fake_score"], confidence["confidence_score"], valid_frames=0)
        self.assertEqual("RETRY", result["decision"])

    def test_blur_reduces_confidence(self):
        clear = self._confidence(0.2, blur=0.0)
        blurry = self._confidence(0.2, blur=0.9)
        self.assertLess(blurry["confidence_score"], clear["confidence_score"])

    def test_low_face_detection_reduces_confidence(self):
        good = self._confidence(0.2, face_frames=100)
        poor = self._confidence(0.2, face_frames=10)
        self.assertLess(poor["confidence_score"], good["confidence_score"])

    def test_output_is_json_serializable(self):
        aggregation = aggregate_frame_scores([0.9] * 100)
        confidence = self._confidence(0.9)
        risk = assess_risk(aggregation["video_fake_score"], confidence["confidence_score"])
        payload = build_output(aggregation, confidence, risk, 100, 100, 100, 0.1, 0.5, 642)
        decoded = json.loads(dumps_output(payload))
        self.assertEqual("BLOCK", decoded["risk"]["decision"])


if __name__ == "__main__":
    unittest.main()
