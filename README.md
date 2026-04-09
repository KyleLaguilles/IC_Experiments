# InceptionContext

A hybrid HAR architecture combining ICGNet's multi-scale inception 
branches with DeepConvContext's cross-window BiLSTM context, designed for sport-specific IMU activity recognition.

## Motivation
Running and logging experiments through commit history; also logged 3 seed experiments for each version on separate Google Sheets 

## Results on Hang-Time HAR
| Model | Macro F1 |
|---|---|
| DeepConvLSTM | 35.7% |
| DCC (Bi) | 48.8% |
| InceptionContext (Version 1) | 50.6% |

## Planned modifications
1. Global statistics branch — explicitly compute mean and variance of the raw signal across the window and concatenate to the GRU output before Stage 2. Directly addresses the standing problem by giving the model an explicit "this window is flat" signal that attention throws away.
2. Equal or reweighted branch filters — currently 32 64 64 64. Try 64 64 64 64 (equal) or even 32 64 64 128 (more weight to larger kernels) since basketball activities have longer timescales than typical ADL datasets.
3. Second GRU layer in Stage 1 — we established the bottleneck is upstream in Stage 1, not in the context LSTM. Adding depth to the within-window GRU specifically might help without the downsides we saw from adding context LSTM layers.
4. Strided conv instead of no pooling — right now we just removed MaxPool entirely. A strided depthwise conv (like TinierHAR uses) could reduce temporal dimension more gracefully than MaxPool while preserving more information.
5. Multi-scale GRU — run parallel GRUs on the inception output at different temporal resolutions and concatenate, rather than a single GRU. Captures both fine and coarse temporal dynamics within a window.
