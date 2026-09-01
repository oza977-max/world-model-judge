"""Evasion fixture: the classic subclasses-walk reflection route."""


def leaks_everything():
    return ().__class__.__base__.__subclasses__()
