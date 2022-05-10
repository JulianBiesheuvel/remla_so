from transform_data import my_bag_of_words

def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    wrong_count = 0
    for ex, ans in zip(examples, answers):
        assert (my_bag_of_words(ex, words_to_index, 4) == ans).all(), "Wrong answer for the case: '%s'" % ex