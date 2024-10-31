from variational_algorithms.use_cases import Instance

class UseCaseInstance(Instance):
    def __init__(self):
        pass

    @property
    def upper_bound(self):
        raise NotImplementedError()

    @property
    def lower_bound(self):
        raise NotImplementedError()

    def export_to_file(self, filename: str):
        raise NotImplementedError()

    def import_from_file(self, filename: str):
        raise NotImplementedError()
