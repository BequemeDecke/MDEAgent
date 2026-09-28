from pathlib import Path
from unittest import TestCase

from mdeagent.util import real_to_virtual


class TestPathMapping(TestCase):
    def test_real_to_virtual(self):
        real_root = Path("/home/user/workspace")
        virtual_root = Path("/workspace")

        # Test case 1: Path is within the real root
        real_path = real_root / "subdir" / "file.txt"
        expected_virtual_path = virtual_root / "subdir" / "file.txt"
        actual_path = real_to_virtual(real_path, real_root, virtual_root)

        self.assertEqual(
            actual_path, expected_virtual_path
        )
