from pathlib import Path
from unittest import TestCase

from mdeagent.util import real_to_virtual, virtual_to_real


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

    def test_virtual_to_real(self):
        real_root = Path("/home/user/workspace")
        virtual_root = Path("/workspace")

        # Test case 2: Path is within the virtual root
        virtual_path = virtual_root / "subdir" / "file.txt"
        expected_real_path = real_root / "subdir" / "file.txt"
        actual_path = virtual_to_real(virtual_path, virtual_root, real_root)

        self.assertEqual(
            actual_path, expected_real_path
        )
