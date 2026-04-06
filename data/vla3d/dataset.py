

class Scene:
    def __init__(self, path: Path):
        self.path = path
        self.scene_id = path.name

    def load_scene_graph(self) -> SceneGraph:
        ...

    def load_statements(self) -> list[ReferentialStatement]:
        ...

    def load_pointcloud(self):
        ...

class VLA3DDataset:
    def __init__(self, root: Path):
        self.root = root

    def scenes(self):
        return [Scene(p) for p in self.root.iterdir() if p.is_dir()]

    def get_scene(self, scene_id: str) -> Scene:
        return Scene(self.root / scene_id)
    
class VLA3D:
    def __init__(self, root: Path):
        self.root = root

    def datasets(self):
        return {
            "3rscan": VLA3DDataset(self.root / "3RScan"),
            "hm3d": VLA3DDataset(self.root / "HM3D"),
        }