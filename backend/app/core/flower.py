from app.core.data_instance import DataInstance


class Flower(DataInstance):

    def __init__(self,
                 uid: int,
                 features: any,
                 target: float) -> None:
        super().__init__(uid, features, target)
