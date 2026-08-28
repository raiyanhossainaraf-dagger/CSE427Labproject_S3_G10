import unittest
from src.preprocessing import get_answer_type, process_qasper_to_tables, normalize_text

class TestPreprocessing(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("  hello   world  "), "hello world")
        self.assertEqual(normalize_text("\nhello\tworld\r"), "hello world")

    def test_get_answer_type(self):
        self.assertEqual(get_answer_type({"unanswerable": True}), "unanswerable")
        self.assertEqual(get_answer_type({"yes_no": True}), "yes")
        self.assertEqual(get_answer_type({"yes_no": False}), "no")
        self.assertEqual(get_answer_type({"extractive_spans": ["span1"]}), "extractive")
        self.assertEqual(get_answer_type({"free_form_answer": "text"}), "abstractive")
        self.assertEqual(get_answer_type({}), "unknown")

    def test_process_qasper_to_tables_small(self):
        ds = {
            "train": {
                "paper1": {
                    "title": "Title 1",
                    "abstract": "Abstract 1",
                    "full_text": [
                        {"section_name": "S1", "paragraphs": ["P1", "P2"]}
                    ],
                    "qas": [
                        {
                            "question_id": "q1",
                            "question": "Q1",
                            "answers": [
                                {
                                    "annotation_id": "ann1",
                                    "answer": {
                                        "extractive_spans": ["P1"],
                                        "evidence": ["P1", "FLOAT SELECTED: fig1"]
                                    }
                                }
                            ]
                        }
                    ],
                    "figures_and_tables": [
                        {"file": "fig1.png", "caption": "Figure 1"}
                    ]
                }
            }
        }
        tables = process_qasper_to_tables(ds)
        self.assertEqual(len(tables["papers"]), 1)
        self.assertEqual(len(tables["sections"]), 2) # Abstract + S1
        self.assertEqual(len(tables["paragraphs"]), 3) # Abstract para + P1 + P2
        self.assertEqual(len(tables["questions"]), 1)
        self.assertEqual(len(tables["answers"]), 1)
        self.assertEqual(len(tables["evidence"]), 2)
        self.assertEqual(len(tables["figures_tables"]), 1)
        
        self.assertEqual(tables["evidence"].iloc[0]["evidence_type"], "text")
        self.assertEqual(tables["evidence"].iloc[1]["evidence_type"], "figure_table")

if __name__ == "__main__":
    unittest.main()
