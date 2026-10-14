"""The suite's own guard: no test reads Claude Code's real transcripts or touches ccdrift's real files."""

from ccdrift.logs import default_source
from ccdrift.state import ccdrift_home


def test_every_test_reads_transcripts_and_keeps_ccdrifts_files_in_folders_of_its_own(tmp_path):
    assert default_source().is_relative_to(tmp_path)
    assert ccdrift_home().is_relative_to(tmp_path)
