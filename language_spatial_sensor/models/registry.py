from typing import Any, Callable, Dict, TypeVar, Union

T = TypeVar("T")


class Registry:
    """Decorator-based registry mapping string keys to classes or callables.

    Usage::

        MY_REGISTRY = Registry("my_thing")

        @MY_REGISTRY.register("foo")
        class Foo(nn.Module): ...

        obj = MY_REGISTRY.build("foo", cfg)

        # Also works for plain functions (e.g. loss functions):
        @MY_REGISTRY.register("bar_loss")
        def bar_loss(pred, target, **kwargs): ...

        result = MY_REGISTRY.build("bar_loss", pred, target)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Union[type, Callable]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(fn: T) -> T:
            if name in self._registry:
                raise KeyError(
                    f"{self._name} registry already contains '{name}'. "
                    f"Existing keys: {list(self._registry)}"
                )
            self._registry[name] = fn  # type: ignore[assignment]
            return fn

        return decorator

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self._name} registry. "
                f"Available: {list(self._registry)}"
            )
        return self._registry[name](*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def keys(self):
        return self._registry.keys()

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={list(self._registry)})"


# Module-level registries — import these everywhere instead of instantiating new ones.
BACKBONE_REGISTRY = Registry("backbone")   # spatial refinement backbone (VisTA, DiT, …)
POOLING_REGISTRY  = Registry("pooling")    # sequence → fixed-dim vector
HEAD_REGISTRY     = Registry("head")       # distribution regression head
LOSS_REGISTRY     = Registry("loss")       # loss functions: (pred, batch_fields…) → scalar
