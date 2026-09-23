"""Alert wording: one title per alert kind, all of them in texts.py and named in the README."""

import re
from pathlib import Path

from ccdrift.check import LOG_ONLY
from ccdrift.texts import ALERT_TITLES


def test_the_alert_kinds_with_a_title_are_exactly_the_ones_the_readme_names():
    # The README's --exec paragraph is where a user learns the values CCDRIFT_ALERT takes,
    # and an alert kind added without it there is one a script can't know to expect.
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    paragraph = readme.split("`--exec` runs a command")[1].split("For example")[0]
    assert set(re.findall(r"`([a-z_]+)`", paragraph)) == set(ALERT_TITLES)
    assert LOG_ONLY <= set(ALERT_TITLES)
