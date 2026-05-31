"""Command line entry point tests."""

import pytest

from ewald.app.cli import main


def test_cli_rejects_missing_project_path(tmp_path, capsys):
    missing_project = tmp_path / "missing.ewld"

    with pytest.raises(SystemExit) as error:
        main([str(missing_project)])

    assert error.value.code == 2
    assert "project file does not exist" in capsys.readouterr().err
