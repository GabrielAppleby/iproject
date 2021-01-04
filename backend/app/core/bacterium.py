from app.core.data_instance import DataInstance


class Bacterium(DataInstance):

    def __init__(self,
                 uid: int,
                 features: any,
                 target: float,
                 umap_x: float,
                 umap_y: float) -> None:
        super().__init__(uid, features, target, umap_x, umap_y)
