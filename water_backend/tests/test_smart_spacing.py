from app.services import smart_spacing as ss
from app.services.text_buffer import TextBuffer


def test_splits_runs_made_of_known_words() -> None:
    assert ss.split_run("HELLOMYNAMEIS") == ["HELLO", "MY", "NAME", "IS"]
    assert ss.split_run("IHAVEAHEADACHE") == ["I", "HAVE", "A", "HEADACHE"]
    assert ss.split_run("whereisthebathroom") == ["WHERE", "IS", "THE", "BATHROOM"]


def test_names_and_known_words_stay_whole() -> None:
    for run in ("JONATHAN", "OLVERA", "CARLOS", "MAYA", "ANDY", "HELLO", "HI"):
        assert ss.split_run(run) == [run]


def test_name_after_an_introducing_word() -> None:
    assert ss.split_run("MYNAMEISJON") == ["MY", "NAME", "IS", "JON"]
    assert ss.split_run("IAMOLVERA") == ["I", "AM", "OLVERA"]


def test_added_words_join_the_vocabulary() -> None:
    assert ss.split_run("ALLIGATORHURTS") == ["ALLIGATORHURTS"]
    ss.add_words(["alligator"])
    assert ss.split_run("ALLIGATORHURTS") == ["ALLIGATOR", "HURTS"]


def test_buffer_split_keeps_one_item_per_letter() -> None:
    t = TextBuffer()
    t.add_word("hello", 0.0)
    for c in "IAMHERE":
        t.add_letter(c, 1.0, manual=True)
    assert t.split_trailing_run(ss.split_run)
    t.add_space(2.0)
    assert t.text == "HELLO I AM HERE"
    t.delete_last()  # the space
    t.delete_last()  # the E
    assert t.text == "HELLO I AM HER"
    assert t.trailing_letters() == "HER"


def test_buffer_split_noop_for_names() -> None:
    t = TextBuffer()
    for c in "JON":
        t.add_letter(c, 0.0, manual=True)
    assert not t.split_trailing_run(ss.split_run)
    assert t.text == "JON"
