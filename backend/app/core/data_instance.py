class DataInstance:

    def __init__(self,
                 uid: int,
                 features: any,
                 target: float) -> None:
        self.uid = uid
        self.features = features
        self.target = target
        super().__init__()
