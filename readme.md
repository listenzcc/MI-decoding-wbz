# MI decoding with EEG

Decoding MI EEG data with MNE[^mne] software.
[^mne]:<https://mne.tools/stable/index.html>

---
[toc]

## Development Diary

### 2025-8-14

- Initialize the project.
- Create 0.overview.py for data overview and plotting
- Implement 1.compute.tfr.py for computing time-frequency representations
- Add 2.plot.erd.exampleChannels.py for plotting ERD with example channels
- Summarize FBCSP voting results in 3.1.mvpa.fbcsp.result.py
- Implement FBCSP decoding using probability voting in 3.mvpa.fbcsp.vote.py
- Define frequency bands in util/bands.py
- Implement data handling in util/data.py
