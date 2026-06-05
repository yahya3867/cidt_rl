from __future__ import annotations

import unittest

from src.slm_client import parse_slm_json


class SlmClientTests(unittest.TestCase):
    def test_parse_plain_json(self) -> None:
        parsed = parse_slm_json('{"summary":"x","recommendation":"y"}')
        self.assertEqual(parsed["summary"], "x")

    def test_parse_fenced_json(self) -> None:
        parsed = parse_slm_json('```json\n{"summary":"x"}\n```')
        self.assertEqual(parsed["summary"], "x")


if __name__ == "__main__":
    unittest.main()
