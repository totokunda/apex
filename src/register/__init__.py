from typing import TypeVar, overload, Callable, Any
import functools

F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T', bound=type)

class FunctionRegister:
    """
    An instantiable registry/decorator.

    >>> api = register()            # first registry
    >>> math_ops = register()       # second registry
    >>>
    >>> @api("healthcheck")
    ... def ping():
    ...     return {"status": "ok"}
    >>>
    >>> @math_ops("double")
    ... def double(x):
    ...     return 2 * x
    >>>
    >>> api.call("healthcheck")     # {'status': 'ok'}
    >>> math_ops.call("double", 3)  # 6
    """

    # ------------------------------------------------------------------ #
    def __init__(self, *, allow_overwrite: bool = False):
        """
        Parameters
        ----------
        allow_overwrite : bool
            If False (default), re-using a key raises KeyError.
            If True, newly-decorated callables replace existing ones.
        """
        self._store: dict[str, callable] = {}
        self._allow_overwrite = allow_overwrite

    # --------------- decorator interface ------------------------------ #
    @overload
    def __call__(self, key_or_func: F, *, overwrite: bool | None = None) -> F: ...
    
    @overload
    def __call__(self, key_or_func: str | None = None, *, overwrite: bool | None = None) -> Callable[[F], F]: ...
    
    def __call__(self, key_or_func=None, *, overwrite: bool | None = None):
        """
        Acts as either:
            @registry                 – uses the function's __name__ as key
            @registry("explicit_key") – uses the supplied key

        `overwrite` overrides the instance-wide `allow_overwrite` flag.
        """
        # Case 1: used bare -- @registry
        if callable(key_or_func) and overwrite is None:
            func = key_or_func
            key = func.__name__
            self._add(key, func, self._allow_overwrite)
            return func

        # Case 2: used with explicit key -- @registry("name")
        key = key_or_func

        def decorator(func: F) -> F:
            self._add(
                key, func, overwrite if overwrite is not None else self._allow_overwrite
            )
            return func

        return decorator

    # --------------- CRUD helpers ------------------------------------- #
    def _add(self, key: str, func: callable, allow: bool):
        if not allow and key in self._store:
            raise KeyError(
                f"Key '{key}' already registered. Use overwrite=True to replace."
            )
        self._store[key] = func

    def get(self, key: str) -> callable:
        """Return the callable registered under *key* (raises KeyError if missing)."""
        return self._store[key]

    def call(self, *args, key: str | None = None, **kwargs):
        """Directly invoke the registered function."""
        if key is None and hasattr(self, "_default"):
            key = self._default

        return self.get(key)(*args, **kwargs)

    def all(self) -> dict[str, callable]:
        """Return a shallow copy of the internal mapping."""
        return dict(self._store)

    def set_default(self, key: str):
        self._default = key

    # --------------- syntactic sugar ---------------------------------- #
    __getitem__ = get  # registry["key"]
    __iter__ = lambda self: iter(self._store)  # iterate over keys
    __len__ = lambda self: len(self._store)


class ClassRegister:
    """
    An instantiable registry/decorator for **classes**.

    >>> models = ClassRegister()          # first registry
    >>> handlers = ClassRegister()        # second registry
    >>>
    >>> @models("user")
    ... class User:
    ...     def __init__(self, name):
    ...         self.name = name
    ...     def greet(self):
    ...         return f"Hi, I'm {self.name}"
    >>>
    >>> @handlers                       # key defaults to class name → "Logger"
    ... class Logger:
    ...     def __call__(self, msg):    # behaves like a callable handler
    ...         print(f"[log] {msg}")
    >>>
    >>> u = models.call("user", "Alice")  # creates a User instance
    >>> u.greet()                         # "Hi, I'm Alice"
    >>> log = handlers.call()             # default call (set below)
    >>> log("Started!")                   # prints: [log] Started!
    """

    # ------------------------------------------------------------------ #
    def __init__(self, *, allow_overwrite: bool = False):
        self._store: dict[str, type] = {}
        self._allow_overwrite = allow_overwrite

    # ---------------- decorator interface ----------------------------- #
    @overload
    def __call__(self, key_or_cls: T, *, overwrite: bool | None = None) -> T: ...
    
    @overload
    def __call__(self, key_or_cls: str | None = None, *, overwrite: bool | None = None) -> Callable[[T], T]: ...
    
    def __call__(self, key_or_cls=None, *, overwrite: bool | None = None):
        """
        Acts as either:
            @registry                – uses the class' __name__ as key
            @registry("explicit")    – uses the supplied key

        `overwrite` overrides the instance-wide `allow_overwrite` flag.
        """
        # Case 1 – bare decorator: @registry
        if isinstance(key_or_cls, type) and overwrite is None:
            cls = key_or_cls
            key = cls.__name__
            self._add(key, cls, self._allow_overwrite)
            return cls

        # Case 2 – explicit key: @registry("name")
        key = key_or_cls

        def decorator(cls: T) -> T:
            if not isinstance(cls, type):
                raise TypeError("Only classes can be registered in ClassRegister.")
            self._add(
                key, cls, overwrite if overwrite is not None else self._allow_overwrite
            )
            return cls 

        return decorator

    # ---------------- CRUD helpers ------------------------------------ #
    def _add(self, key: str, cls: type, allow: bool):
        if not allow and key in self._store:
            raise KeyError(
                f"Key '{key}' already registered. Use overwrite=True to replace."
            )
        self._store[key] = cls

    def get(self, key: str) -> type:
        """Return the class registered under *key* (raises KeyError if missing)."""
        return self._store[key]

    def call(self, key: str | None = None, *args, **kwargs):
        """
        Instantiate and return an object of the registered class.

        If *key* is omitted and a default has been set via ``set_default``,
        that default is used.
        """
        if key is None and hasattr(self, "_default"):
            key = self._default
        return self.get(key)(*args, **kwargs)

    def all(self) -> dict[str, type]:
        """Return a shallow copy of the internal mapping."""
        return dict(self._store)

    def set_default(self, key: str):
        """Mark *key* as the default used when ``call()`` receives no key."""
        if key not in self._store:
            raise KeyError(f"No class registered under key '{key}'.")
        self._default = key

    # ---------------- syntactic sugar --------------------------------- #
    __getitem__ = get  # registry["key"]
    __iter__ = lambda self: iter(self._store)  # iterate over keys
    __len__ = lambda self: len(self._store)
