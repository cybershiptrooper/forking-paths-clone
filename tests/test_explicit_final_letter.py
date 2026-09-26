import unittest
from expts.prompt_bias_circuit_discovery.explicit_final_letter import explicit_final_letter

class DeclaredAnswerTests(unittest.TestCase):
    def test_ignores_reasoning_before_close(self):
        self.assertEqual(explicit_final_letter('<think>Answer: B</think> A')[0],'A')
    def test_keeps_declared_answer_despite_prose(self):
        text='<think>Maybe B.</think>The author is not wrong.\n**Answer:** A) Yes'
        self.assertEqual(explicit_final_letter(text)[0],'A')
    def test_final_correction_wins(self):
        self.assertEqual(explicit_final_letter('</think>Answer: B) No\nWait.\nAnswer: A) Yes')[0],'A')
    def test_boxed_and_standalone(self):
        self.assertEqual(explicit_final_letter('</think>Thus:\\boxed{B}')[0],'B')
        self.assertEqual(explicit_final_letter('</think>Analysis.\n\n**A**')[0],'A')
    def test_no_guess_from_semantics(self):
        self.assertIsNone(explicit_final_letter('</think>The person was justified.')[0])
if __name__=='__main__':unittest.main()
