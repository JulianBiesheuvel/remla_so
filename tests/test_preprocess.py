from code.preprocess_data import text_prepare

# prepared_questions = []
# for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
#     line = text_prepare(line.strip())
#     prepared_questions.append(line)
# text_prepare_results = '\n'.join(prepared_questions)


def test_text_prepare():
    examples = [
        "SQL Server - any equivalent of Excel's CHOOSE function?",
        "How to free c++ memory vector<int> * arr?",
    ]
    answers = [
        "sql server equivalent excels choose function",
        "free c++ memory vectorint arr",
    ]
    for ex, ans in zip(examples, answers):
        assert text_prepare(ex) == ans, "Wrong answer for the case: '%s'" % ex
