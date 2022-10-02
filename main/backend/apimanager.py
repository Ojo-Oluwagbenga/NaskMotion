import importlib
from . import api

class Manager:
    def manager(response, apiclass, apimethod):
        print("Calles")
        class_ = getattr(api, apiclass)

        return getattr(class_, apimethod)(class_, response)
