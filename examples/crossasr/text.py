import functools

@functools.total_ordering
class Text:
    def __init__(self, id: int, text: str):
        self.id = id
        self.text = text

    def __eq__(self, other):
        return self.id == other.id and self.text == other.text

    def __lt__(self, other):
        return (self.id, self.text) < (other.id, other.text)

    def getId(self):
        return self.id

    def setId(self, id: int):
        self.id = id

    def getText(self):
        return self.text

    def setText(self, text: str):
        self.text = text
