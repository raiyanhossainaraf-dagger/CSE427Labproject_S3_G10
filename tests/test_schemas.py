import unittest
import pandas as pd
from src.schemas import (
    validate_table, generate_section_id, generate_paragraph_id,
    generate_evidence_id, generate_float_id, SPLITS, ANSWER_TYPES, EVIDENCE_TYPES
)

class TestSchemas(unittest.TestCase):
    def test_stable_ids(self):
        paper_id = "123"
        sec_id = generate_section_id(paper_id, "full_text", 1)
        self.assertEqual(sec_id, "123::section::full_text::1")
        
        para_id = generate_paragraph_id(sec_id, 2)
        self.assertEqual(para_id, "123::section::full_text::1::paragraph::2")
        
        ann_id = "ann456"
        ev_id = generate_evidence_id(ann_id, 0)
        self.assertEqual(ev_id, "ann456::evidence::0")
        
        float_id = generate_float_id(paper_id, 5)
        self.assertEqual(float_id, "123::float::5")

    def test_validate_table_valid(self):
        df = pd.DataFrame({
            "paper_id": ["p1", "p2"],
            "split": ["train", "validation"],
            "title": ["T1", "T2"],
            "abstract": ["A1", "A2"],
            "num_full_text_sections": [1, 2],
            "num_source_sections": [2, 3],
            "num_paragraphs": [10, 20],
            "num_questions": [5, 6],
            "full_text_word_count": [1000, 2000]
        })
        validate_table(df, "papers")

    def test_validate_table_invalid_split(self):
        df = pd.DataFrame({
            "paper_id": ["p1"],
            "split": ["invalid_split"],
            "title": ["T1"],
            "abstract": ["A1"],
            "num_full_text_sections": [1],
            "num_source_sections": [2],
            "num_paragraphs": [10],
            "num_questions": [5],
            "full_text_word_count": [1000]
        })
        with self.assertRaises(ValueError):
            validate_table(df, "papers")

    def test_validate_table_duplicate_id(self):
        df = pd.DataFrame({
            "paper_id": ["p1", "p1"],
            "split": ["train", "train"],
            "title": ["T1", "T1"],
            "abstract": ["A1", "A1"],
            "num_full_text_sections": [1, 1],
            "num_source_sections": [2, 2],
            "num_paragraphs": [10, 10],
            "num_questions": [5, 5],
            "full_text_word_count": [1000, 1000]
        })
        with self.assertRaises(ValueError):
            validate_table(df, "papers")

if __name__ == "__main__":
    unittest.main()
