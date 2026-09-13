import unittest

from easyvolcap.utils.custom_indexer.sparse_topk_indexer import _needs_pending_topk_indices_buf


class SparseTopkWorkspaceTests(unittest.TestCase):
    def test_merge_index_kernel_skips_unused_pending_index_buffer(self):
        self.assertFalse(_needs_pending_topk_indices_buf(use_merge_index_kernel=True))
        self.assertTrue(_needs_pending_topk_indices_buf(use_merge_index_kernel=False))


if __name__ == "__main__":
    unittest.main()
