"""Model Behavior Lab: reproducible experiments on whether AI services behave as configured."""

__version__ = "0.2.0"

REPOSITORY = "https://github.com/cynkai/model-behavior-lab"


def install_hint(extra: str, needs: str) -> str:
    """How to add an optional extra (``api``, ``dashboard``).

    Never a bare ``pip install model-behavior-lab[...]``: the project is not on PyPI,
    so that command would install whatever package someone publishes under the name.
    """
    return (
        f'{needs}. In your clone of the repository run: pip install -e ".[{extra}]" '
        f"(with uv: uv sync --extra {extra}, and add --extra for every other extra you use, "
        "because uv sync removes extras it is not given). Without a clone: "
        f'pip install "model-behavior-lab[{extra}] @ git+{REPOSITORY}". '
        "model-behavior-lab is not published on PyPI; a package of that name there is not "
        "this project"
    )
