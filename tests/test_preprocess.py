from src.preprocess import preprocess


def test_text_prepare() -> None:
    examples = [
        "SQL Server - any equivalent of Excel's CHOOSE function?",
        "How to free c++ memory vector<int> * arr?",
    ]
    answers = [
        "sql server equivalent excels choose function",
        "free c++ memory vectorint arr",
    ]
    for output, expected in zip(preprocess(examples), answers):
        assert output == expected, "Wrong answer for the case: '%s'" % expected
