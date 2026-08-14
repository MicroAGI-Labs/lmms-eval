import pickle


class PickleWriteable:
    """Mixin for persisting an instance with pickle."""

    def save(self, path):
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except (pickle.PickleError, OSError) as e:
            raise OSError(f"Unable to save {self.__class__.__name__} to path: {path}") from e

    @classmethod
    def load(cls, path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, OSError) as e:
            raise OSError(f"Unable to load {cls.__name__} from path: {path}") from e
