# Findings so far

From one developer's Claude Code logs, Aug 6 - Sep 15, 2026. Aggregate numbers only.

- **Cache: a real signal.** A Claude Code caching regression (versions 2.1.233-2.1.258, in use Aug 16 - Sep 4) is flagged from Aug 18. Its first days score z = -3.8, -5.4, -2.8, -3.2 against a cutoff of 3.0, so a flag takes 3 deviant days among any 4: once Claude Code's 30-day transcript cleanup deleted part of Aug 15-16, requiring 3 in a row missed it. With the incident left out, a 5% drop is caught from all 3 starting days in the 12 clean days that remain.
- **Haiku: a real signal on the main thread**, which never uses Haiku, so a 5% shift is caught from every starting day. Subagent Haiku bursts make the all-turn share noisy: on 5 synthetic logs it was falsely flagged on 3 (on 1 when the deviant days had to be in a row), the main thread on none.
- **Effort: not detectable from one user's logs.** Over the 12 clean days its daily level swung between 0.33 and 0.67, more than a 70% cut in thinking tokens moves it, so no size of drop was caught, on all turns or the main thread. Detecting it points to pooled data from many users.

[what-ccdrift-caught.html](what-ccdrift-caught.html) charts the regression and the detection results.
