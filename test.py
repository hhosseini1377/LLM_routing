class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def say_hello(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

def UpdateClass(cls, **kwargs):
    cls.salam = 'salam'

    return cls

Person = UpdateClass(Person, name="John", age=30)

if __name__ == "__main__":
    person = Person("John", 30)
    person.say_hello()
    print(person.salam)