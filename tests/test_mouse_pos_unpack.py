from unittest.mock import MagicMock


def test_mouse_pos_handling_tuple():
    # Test when io.MousePos is a tuple
    mock_io = MagicMock()
    mock_io.MousePos = (100.0, 200.0)

    m_pos = mock_io.MousePos
    if hasattr(m_pos, "x"):
        mouse_pos = (float(m_pos.x), float(m_pos.y))
    else:
        mouse_pos = (float(m_pos[0]), float(m_pos[1]))

    assert mouse_pos == (100.0, 200.0)


def test_mouse_pos_handling_object():
    # Test when io.MousePos is an object with x, y attributes
    class Vec2:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    mock_io = MagicMock()
    mock_io.MousePos = Vec2(150.0, 250.0)

    m_pos = mock_io.MousePos
    if hasattr(m_pos, "x"):
        mouse_pos = (float(m_pos.x), float(m_pos.y))
    else:
        mouse_pos = (float(m_pos[0]), float(m_pos[1]))

    assert mouse_pos == (150.0, 250.0)
