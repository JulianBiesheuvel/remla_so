from src.model import TFIDF, BagOfWords


def test_my_bag_of_words() -> None:
    emb = BagOfWords(size=4)
    emb.fit(["hi", "hi", "hi", "hi", "you", "you", "you", "me", "me", "are"])
    # words_to_index = {"hi": 0, "you": 1, "me": 2, "are": 3}
    examples = ["hi how are you"]
    answers = [[1, 1, 0, 1]]
    for ex, expected in zip(examples, answers):
        assert (emb._transform(ex) == expected).all(), (
            "Wrong answer for the case: '%s'" % expected
        )


def test_tfidf() -> None:
    emb = TFIDF(min_df=0.01, max_df=1.0)

    data = [
        "sql server equivalent excels choose function",
        "free c++ memory vectorint arr",
    ]

    o1 = emb.fit_transform(data)

    o2 = emb.transform(data)

    assert (o1 != o2).nnz == 0
