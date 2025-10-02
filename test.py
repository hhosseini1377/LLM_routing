class ShirinMixin:
    def __init__(self):
        self.ahmad = 'ahmad'
        print('mixin init')
    @classmethod
    def create_class(cls):
        print(cls)
        print('salam')

    def print_ahmad(self):
        print('salam')
    
    def call_ahmad(self):
        print(self.print_ahmad())

class TaghiMixin:
    def call_ahmad(self):
        print('salam taghi')
    
class Embedding( TaghiMixin, ShirinMixin):
    def __init__(self):
        super().__init__()
        self.mohsen = 'mohsen'
    
    def print_ahmad(self):
        print('salam dadash')

embedding = Embedding()
embedding.call_ahmad()
