from relevance_scorer import calculate_grounded_accuracy


class MockDocument:
    def __init__(self, text):
        self.page_content = text


context = MockDocument(
    """
    Cyber stalking means repeatedly contacting a person
    despite a clear indication of disinterest.

    It also includes monitoring electronic communication
    and watching or spying on a person in a manner that
    causes fear, alarm or distress.

    A person convicted of cyber stalking may be punished
    with imprisonment up to three years or fine up to
    one million rupees or both.
    """
)


answer = """
Cyber stalking includes repeatedly contacting a person
despite a clear indication of disinterest.

It may also include monitoring electronic communication
or watching and spying in a way that causes fear,
alarm or distress.

The punishment may include imprisonment up to three years
or a fine up to one million rupees or both.
"""


score = calculate_grounded_accuracy(
    answer,
    [context]
)

print("=" * 50)
print("GROUNDED ACCURACY TEST")
print("=" * 50)
print("Accuracy:", score, "%")