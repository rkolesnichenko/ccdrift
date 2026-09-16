# Findings so far

From one developer's Claude Code logs, Aug 6 - Sep 15, 2026. Aggregate numbers only.

- **Cache: a real signal.** A Claude Code caching regression (versions 2.1.233-2.1.258, in use Aug 16 - Sep 4) is flagged from Aug 18. Its first days score z = -3.8, -5.4, -2.8, -3.2 against a cutoff of 3.0, so a flag takes 3 deviant days among any 4: once Claude Code's 30-day transcript cleanup deleted part of Aug 15-16, requiring 3 in a row missed it. With the incident left out, a 5% drop is caught from all 3 starting days in the 12 clean days that remain.
- **Following an incident.** Judged against the 14 days before its first flagged day (Aug 6, 7, 15, 16, 17), every day of the caching regression from Aug 18 to Sep 1 scores z ≤ −3.1. Against a rolling baseline z was back to about 0 by Aug 26, while 5–10% of prompt turns still missed. Pooling 3 days, the incident is back to normal from Sep 4 and closes on Sep 6; 2.1.259 was first used on Sep 3 and 2.1.260 on Sep 4. Judging single days instead closed it on Sep 4 "from Sep 2", though Sep 3 still had a miss. Missed prompt turns wrote 17.6M tokens to the cache from Aug 16 to Sep 4.
- **Haiku: a real signal on the main thread**, which never uses Haiku, so a 5% shift is caught from every starting day. Subagent Haiku bursts make the all-turn share noisy: on 5 synthetic logs it was falsely flagged on 3 (on 1 when the deviant days had to be in a row), the main thread on none.
- **Effort: not detectable from one user's logs.** Over the 12 clean days its daily level swung between 0.33 and 0.67, more than a 70% cut in thinking tokens moves it, so no size of drop was caught, on all turns or the main thread. Detecting it points to pooled data from many users.

[what-ccdrift-caught.html](what-ccdrift-caught.html) charts the regression and the detection results.

Reproduce them with the harness in `lab/`; see [lab/README.md](../lab/README.md).
