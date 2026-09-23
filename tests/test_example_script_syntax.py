import py_compile
import os


def test_bags_collision_script_syntax():
    script_path = os.path.abspath("examples/bags_collision.py")
    assert os.path.exists(script_path), "bags_collision.py must exist"

    # Verify that the script compiles cleanly without SyntaxError
    py_compile.compile(script_path, doraise=True)


def test_cell_division_script_syntax():
    script_path = os.path.abspath("examples/cell_division.py")
    assert os.path.exists(script_path), "cell_division.py must exist"

    # Verify that the script compiles cleanly without SyntaxError
    py_compile.compile(script_path, doraise=True)

