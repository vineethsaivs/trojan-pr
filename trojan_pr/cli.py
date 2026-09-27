"""python -m trojan_pr.cli verify|keygen|hist ... (runs plumbline.cli, the working package name)."""
import runpy

if __name__ == "__main__":
    runpy.run_module("plumbline.cli", run_name="__main__")
