import unittest
import pandas as pd
from src.chunking import chunk_paragraphs, generate_chunk_id

import os

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.papers_df = pd.DataFrame({
            "paper_id": ["p1"],
            "title": ["A very long title for testing purposes"]
        })
        self.paragraphs_df = pd.DataFrame({
            "paper_id": ["p1", "p1", "p1"],
            "section_id": ["p1::section::full_text::0", "p1::section::full_text::0", "p1::section::full_text::0"],
            "section_name": ["Introduction", "Introduction", "Introduction"],
            "paragraph_id": ["p1::section::full_text::0::paragraph::0", "p1::section::full_text::0::paragraph::1", "p1::section::full_text::0::paragraph::2"],
            "text": ["Short para 1", "Short para 2", "This is a much longer paragraph that should ideally be split if we set the token limit very low for testing purposes."],
            "split": ["train", "train", "train"]
        })

    def test_chunking_grouping(self):
        # Using word-split (mock-like) for simplicity in unit test
        chunks = chunk_paragraphs(self.paragraphs_df, self.papers_df, max_tokens=100, use_hf=False)
        self.assertGreater(len(chunks), 0)
        self.assertEqual(chunks.iloc[0]["paper_id"], "p1")
        # Check if first chunk contains multiple paragraphs if they fit
        if len(chunks.iloc[0]["paragraph_ids"]) > 1:
             self.assertIn("p1::section::full_text::0::paragraph::0", chunks.iloc[0]["paragraph_ids"])
             self.assertIn("p1::section::full_text::0::paragraph::1", chunks.iloc[0]["paragraph_ids"])

    def test_chunking_splitting(self):
        # With use_hf=False, tokenize uses .split()
        # metadata "Title: A very long title for testing purposes\nSection: Introduction\nPassage: "
        # word count: Title(1) : (1) A(1) very(1) long(1) title(1) for(1) testing(1) purposes(1) = 9 words
        # Section(1) : (1) Introduction(1) = 3 words
        # Passage(1) : (1) = 2 words
        # Total metadata approx 14 words.
        # Max tokens 20 -> available 6 words.
        chunks = chunk_paragraphs(self.paragraphs_df, self.papers_df, max_tokens=25, use_hf=False)
        
        partial_chunks = chunks[chunks["is_partial_paragraph"] == True]
        self.assertGreater(len(partial_chunks), 0)
        self.assertEqual(partial_chunks.iloc[0]["parent_paragraph_id"], "p1::section::full_text::0::paragraph::2")
        self.assertIsNotNone(partial_chunks.iloc[0]["paragraph_token_start"])

    def test_deterministic_ids(self):
        chunks1 = chunk_paragraphs(self.paragraphs_df, self.papers_df, max_tokens=50, use_hf=False)
        chunks2 = chunk_paragraphs(self.paragraphs_df, self.papers_df, max_tokens=50, use_hf=False)
        pd.testing.assert_frame_equal(chunks1, chunks2)

    @unittest.skipUnless(os.environ.get("RUN_HF_INTEGRATION") == "1", "Skipping HF integration test")
    def test_hf_integration(self):
        chunks = chunk_paragraphs(self.paragraphs_df, self.papers_df, max_tokens=384, use_hf=True)
        self.assertGreater(len(chunks), 0)

if __name__ == "__main__":
    unittest.main()
