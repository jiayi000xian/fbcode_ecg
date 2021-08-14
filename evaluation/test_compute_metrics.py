import unittest

from .gen_curve import compute_metrics


class TestData:
    def __init__(self):
        self.preds = [
            [[1, 2, 3]],
            [[]],
            [[]],
            [[1, 2, 3]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        ]
        self.golds = [
            [[1, 2, 3]],
            [[]],
            [[1, 2, 3]],
            [[]],
            [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        ]
        self.examples = []
        for pred, gold in zip(self.preds, self.golds):
            self.examples.append([pred, gold])
        self.expected = [
            {
                "fp": 0,
                "tp": 3,
                "fn": 0,
                "micro_precision": 1,
                "micro_f1": 1,
                "micro_recall": 1,
                "P@1": 1,
                "P@3": 1,
                "P@5": 0.6,
            },
            {
                "fp": 0,
                "tp": 0,
                "fn": 0,
                "micro_precision": 1,
                "micro_f1": 1,
                "micro_recall": 1,
                "P@1": 0,
                "P@3": 0,
                "P@5": 0,
            },
            {
                "fp": 0,
                "tp": 0,
                "fn": 3,
                "micro_precision": 0,
                "micro_f1": 0,
                "micro_recall": 0,
                "P@1": 0,
                "P@3": 0,
                "P@5": 0,
            },
            {
                "fp": 3,
                "tp": 0,
                "fn": 0,
                "micro_precision": 0,
                "micro_f1": 0,
                "micro_recall": 0,
                "P@1": 0,
                "P@3": 0,
                "P@5": 0,
            },
            {
                "fp": 0,
                "tp": 10,
                "fn": 0,
                "micro_precision": 1,
                "micro_f1": 1,
                "micro_recall": 1,
                "P@1": 0.75,
                "P@3": 0.75,
                "P@5": 0.75,
            },
        ]


class TestMetricsCompute(unittest.TestCase):
    def setUp(self):
        self.data = TestData()

    def test_compute_metrics(self):
        res = []
        for example in self.data.examples:
            res.append(compute_metrics(example[0], example[1]))

        assert res == self.data.expected
        self.assertListEqual(res, self.data.expected)


if __name__ == "__main__":
    unittest.main()
