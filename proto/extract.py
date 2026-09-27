"""Function-extract replay: load one repo file at a SHA without installing the repo.

Runs INSIDE the sandbox (no network). The control plane ships `files`, a dict
{repo_path: source} fetched at the SHA (see fetch.py). Imports resolve as:
  1. module in `files`            -> exec the real source (same rules, recursively)
  2. repo-package module in SHIMS  -> hand-written single-rank CPU shim
  3. installed top-level package   -> real import (torch, numpy, stdlib)
  4. anything else                 -> stub (identity decorator, falsy, isinstance-safe)
"""
import ast, importlib.util, linecache, sys, types

_ACTIVE = None  # the Loader currently installed (one per sandbox process)


class _StubMeta(type):
    def __call__(cls, *a, **k):
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]                      # @decorator
        return _StubInst()
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub_class(name)
    def __iter__(cls):
        return iter(())


def _stub_class(name):
    return _StubMeta(name, (), {})


class _StubInst:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub_class(name)
    def __call__(self, *a, **k):
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]                      # @decorator(args)
        return _StubInst()
    def __bool__(self):
        return False
    def __iter__(self):
        return iter(())


class StubModule(types.ModuleType):
    __path__ = []                            # lets `import a.b.c` walk through
    __all__ = []                             # star-import from a stub imports nothing
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        full = f"{self.__name__}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        if _ACTIVE and (full in _ACTIVE.shims or _ACTIVE.path_of(full)):
            return importlib.import_module(full)   # `from pkg import submodule` must reach the shim
        return _stub_class(name)


def _module_from_dict(name, d):
    m = types.ModuleType(name)
    m.__dict__.update(d)
    m.__path__ = []
    return m


class Loader:
    def __init__(self, files, repo_root_pkg, shims=None):
        self.files = files                  # {"deepspeed/runtime/utils.py": src}
        self.root = repo_root_pkg           # "deepspeed"
        self.shims = shims or {}            # {"deepspeed.comm": {...}}
        self.stubbed = set()
        self.degraded = []                  # siblings that failed to exec and fell back to stubs
        self.target = None

    def path_of(self, modname):
        base = modname.replace(".", "/")
        for p in (base + ".py", base + "/__init__.py"):
            if p in self.files:
                return p
        return None

    def install(self):
        global _ACTIVE
        _ACTIVE = loader = self
        class Finder:
            def find_spec(self, name, path=None, target=None):
                top = name.split(".")[0]
                if loader.path_of(name) or name in loader.shims:
                    return importlib.util.spec_from_loader(name, _L(loader))
                if top == loader.root or not _installed(top):
                    return importlib.util.spec_from_loader(name, _L(loader))
                return None
        self.finder = Finder()
        sys.meta_path.insert(0, self.finder)

    def uninstall(self):
        sys.meta_path.remove(self.finder)
        for k in list(sys.modules):
            if k.split(".")[0] == self.root or k in self.stubbed:
                del sys.modules[k]

    def load(self, modname):
        self.target = modname
        self.install()
        return importlib.import_module(modname)


class _L:
    def __init__(self, loader):
        self.l = loader
    def create_module(self, spec):
        name = spec.name
        if name in self.l.shims:
            return _module_from_dict(name, self.l.shims[name])
        if self.l.path_of(name):
            m = types.ModuleType(name)
            p = self.l.path_of(name)
            m.__file__ = p
            m.__path__ = [] if p.endswith("__init__.py") else None
            if m.__path__ is None:
                del m.__path__
            m.__package__ = name if p.endswith("__init__.py") else name.rpartition(".")[0]
            return m
        self.l.stubbed.add(name)
        return StubModule(name)
    def exec_module(self, m):
        p = self.l.path_of(m.__name__)
        if p and m.__name__ not in self.l.shims:
            src = self.l.files[p]
            linecache.cache[p] = (len(src), None, src.splitlines(True), p)   # inspect.getsource works
            try:
                exec(compile(src, p, "exec"), m.__dict__)
            except Exception as e:
                if m.__name__ == self.l.target:
                    raise
                m.__class__ = StubModule          # sibling failed: keep what ran, stub the rest
                self.l.stubbed.add(m.__name__)
                self.l.degraded.append(f"{m.__name__}: {type(e).__name__}: {e}")


_inst_cache = {}
def _installed(top):
    if top not in _inst_cache:
        import importlib.machinery as im   # PathFinder directly: our own finder must not recurse
        _inst_cache[top] = (top in sys.modules or top in sys.builtin_module_names
                            or im.PathFinder.find_spec(top) is not None)
    return _inst_cache[top]


def needed_siblings(src, modname, is_pkg=False):
    """Static pre-pass (control plane): which repo modules must be REAL, not stubbed.
    Star imports, ALL_CAPS constants, and base classes of classes defined here."""
    tree = ast.parse(src)
    pkg = modname if is_pkg else modname.rpartition(".")[0]
    bases = {b.id for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
             for b in n.bases if isinstance(b, ast.Name)}
    need = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            if n.level:
                parts = pkg.split(".")
                base = ".".join(parts[: len(parts) - (n.level - 1)])
                mod = f"{base}.{mod}" if mod else base
            names = [a.name for a in n.names]
            if "*" in names or any(x.isupper() for x in names) or bases & set(names):
                need.add(mod)
    return need
