# PRECISE AND SIMPLE AUDIO-TO-SCORE ALIGNMENT

Silvan D. PETER<sup>1,</sup> <sup>2</sup>, Patricia HU (胡紫<sup>漪</sup>) <sup>1</sup>, and Gerhard WIDMER<sup>1,</sup> <sup>2</sup>

<sup>1</sup>Institute of Computational Perception, Johannes Kepler University, Linz, Austria <sup>2</sup>LIT AI Lab, Linz Institute of Technology, Linz, Austria 

## 1. INTRODUCTION

Audio-to-score alignment is a long-standing challenge in music information retrieval and arguably the most widely applicable alignment task for music research. Alignment algorithms match two versions of a piece of music, and for this to work these versions need to be in comparable formats. Audio-to-audio alignment matches audio features; when matching audio files to scores, they must either synthesize the score or derive audio-like features by means of piano rolls or similar feature sequences [1–3]. Symbolic alignment, by contrast, matches symbolically encoded notes; in an audio-to-score scenario these would be obtained by a transcription of the audio file [4, 5]. In this article, we present an algorithm that bridges audiolike and symbol-level features directly. Sequential audio features encoding onset and spectral activation (see Figure 1) are matched to score positions by a bespoke dynamic programming-based matching algorithm derived from symbolic alignment methods. The resulting method is both precise - surpassing widely used audio-to-audio approaches based on synthesized scores -, and remains flexible in its digital signal processing components, i.e., the method is adaptable to diverse timbral characteristics without requiring a separate transcription model. Furthermore it inherits some of the symbolic alignment runtime advantages with an algorithmic complexity that is at worst linear in the length of the (typically short) symbolic score and (typically long) audio feature sequence. In the following sections, we provide a detailed algorithm description and evaluate its alignment quality on a large-scale dataset of solo piano recordings. 

## 2. ALIGNMENT METHOD

## 2.1 Signal Processing

The audio signal is processed into two feature sequences, one for onset (time) information, the other for spectra (pitch) information. As a first step, the stereo signal is summed to mono and then sent through an IIR filterbank of second-order Butterworth filters. The filterbank consists of 88 filters centered at the key frequencies of an equally tempered 440 Hz Chamber pitch piano. The passband limits are set to the quarter tone middle points between adjacen pitch frequencies. The default filterbank is set up for repertoire of this temperament, tuning, and register. Different setups are possible if the musical material is known to differ. The 88 filtered signals are aggregated by window-wise maximal values with window width and hop size being set for 50 Hz frame sequences and stacked as a spectrogram. The onset features are derived by a superflux algorithm [6] from the framed signal: a maximum filtered frame (across three vertically adjacent frequency bins) is subtracted from the subsequent frame and halfwave rectified. The resulting feature is bin-wise normalised to one. The durational feature is directly given by the original filtered and framed signal, again normalized to one for each frequency bin. While this procedure results in usable pitch-wise activation features for piano music, onset activation, normalization, and framerate can be adapted to suit the needs of different types of audio. The score representation consists of a list of score chords (for simplicity just coinciding notes irrespective of durations or voices) given by their onset beat position and the MIDI pitches of the notes encoded at this position. 

## 2.2 Dynamic Programming

The alignment algorithm treats the pitch-wise onset and spectral activations as a proto-transcription and relates it to score information. For any position in the score and each pitch expected to be played at it, all positions in a given temporal window of the framed signal are given a cost related to the best fitting onset position with subsequent spectral activation. In typical dynamic programming fashion it starts at the beginning of both sequences and works through all score and signal window positions while keeping track of previously best aligned subsequences. Pseudocode 1 outlines the algorithm structure. 

The cost function relating score positions to onset times in the spectrogram combines three components: an onset term (strong onset activation), a spectral term (continued spectral energy for several frames), and a stretch term (favoring low time warping or tempo variation). The latter is influenced by a path-wise beat period estimate that is continuously updated, associating lower costs to less locally variable tempo estimates. The algorithm affords several tuning parameters like the stretch limits and the weights to sum the cost from its stretch, onset, and spectral components. Each cost component is normalized to fall between zero and one. 

```csv
Algorithm 1 Score-to-Audio Alignment via Dynamic Programming
Require: score_onset_times[i] ▷ score times in beats
Require: pitch_sets[i] ▷ notated pitches at score onset
Require: onsets[p,t] ▷ onset activation for pitch p at frame t
Require: spec[p,t] ▷ spectral presence for pitch p at frame t
Require: stretch_limits, cost_weights
1: D[i,j] ← ∞ ▷ accumulated cost
2: B[i,j] ← -1 ▷ backpointer
3: BP[i,j] ← bpinit ▷ beat period estimate
4: D[0,0] ← 0
5: for i = 0 to M - 1 do ▷ loop over score onsets
6: for j = 0 to N - 1 do ▷ loop over audio frames
7: if D[i,j] = ∞ then
8: continue
9: end if
10: bp ← BP[i,j] ▷ get local beat period estimate
11: Δscore ← score_onset_times[i + 1] - score_onset_times[i]
12: candidate_frames ← compute_frame_window(j,bp,Δscore,stretch_limits)
13: for all p ∈ pitch_sets[i + 1] do
14: for all j' ∈ candidate_frames do
15: stretch_term ← stretch_cost(j' - j,bp,Δscore)
16: onset_term ← onsets[p,j']
17: spec_term ← mink(spec[p,j' + k])
18: transition_cost ← D[i,j]
+ w_onset · onset_term
+ w_stretch · stretch_term
+ w_spec · spec_term
19: if transition_cost < D[i + 1,j'] then
20: D[i + 1,j'] ← transition_cost
21: B[i + 1,j'] ← j ▷ backtracking pointer
22: BP[i + 1,j'] ← update_beat_period(j' - j,Δscore,bp)
23: end if
24: end for
25: end for
26: end for
27: D[i + 1,mask_cost_above_reset_threshold] ← ∞
28: end for
29: alignment ← backtrack_through_pointers(B)
30: return alignment 
```

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-25/73fa58a2-bf17-4d90-bc65-bb2a47a57407/c71b87e1b37832b9bd4e0d5a0b989e68e83b58e5f05283525a12d3e6ef72a965.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-25/73fa58a2-bf17-4d90-bc65-bb2a47a57407/d52a11229820f814d9b1ffd05f9495f71695cd81274d87582c1d4cd9264b80e1.jpg)



Figure 1. Spectral (left) and onset (right) activation features on the first ten seconds of a piano recording. There are 88 frequency bins (rows) logarithmically spaced and centered on the piano key frequencies. The temporal frame rate is 50 Hz.


<table><tr><td></td><td>Method</td><td>Mean (ms)</td><td>Median (ms)</td><td>&lt; 50 ms (%)</td><td>&lt; 100 ms (%)</td><td>&lt; 200 ms (%)</td><td>&lt; 500 ms (%)</td></tr><tr><td rowspan="3">]</td><td>Audio-to-Audio</td><td>135</td><td>49</td><td>53.2</td><td>74.4</td><td>87.7</td><td>91.7</td></tr><tr><td>Audio-to-Score (ours)</td><td>86</td><td>21</td><td>83.7</td><td>91.7</td><td>95.2</td><td>97.9</td></tr><tr><td>MIDI-to-Score</td><td>6</td><td>0</td><td>98.1</td><td>98.5</td><td>99.2</td><td>99.7</td></tr></table>


Table 1. Alignment results across a baseline audio-to-audio method using a synthesized score and onset as well as chroma features (first row), our proposed mixed audio-symbolic method (second row), and a MIDI-based symbolic alignment method (third row). Mean and median values are given in milliseconds, lower values are better. The remaining columns show the percentage of absolute errors below given thresholds, higher values are better. All values are averaged performance-wise across the dataset.


## 3. EVALUATION

We evaluate our algorithm on over 300 piano performances from the (n)ASAP Dataset [7]. We compare it to an audioto-audio alignment baseline which uses Dynamic Time Warping on both onset-related and spectral features. The implementation is given by the synctoolbox library [8]. Audio-to-audio alignment based on a mix of features and synthesized audio is a common baseline for audio-to-score alignment. When high-quality transcriptions are available, symbolic alignment becomes a more precise baseline. To give an estimated upper bound for the quality of transcription-based symbolic alignment, we assess a MIDI-to-score alignment method (“DualDTWMatcher” ) from the parangonar library [7] using the recorded MIDI performances of the (n)ASAP Dataset as proxies for perfect transcriptions. Table 1 shows the results in terms of mean and median errors as well as percentages of errors below four different thresholds. 

Our method surpasses the baseline audio-to-audio method at all levels of precision, yet unsurprisingly falls short of the precision of a symbolic alignment model. Notably, several alignments were excluded from the audioto-audio version where an obviously spurious alignment was computed, while both our method and the MIDI-toscore alignment worked robustly across the entire dataset. There is a runtime to precision tradeoff in the setting of the window size, frame rate, and threshold for cost rest. The higher these values, the more precise the alignment becomes, and the longer it takes to compute it. The values shown stem from a parameter set on the precise yet slow side (no threshold, medium window, 50 Hz). However, we did not optimize the parameter settings. 

## 4. CONCLUSION

We introduce an audio-to-score algorithm which uses both onset and spectral audio features in a note-based matching procedure typically found in symbolic alignment. Our method leverages dynamic beat period estimates and score-informed pitch-wise onset and spectral processing to produce highly precise alignments. It relies on standard digital signal processing and dynamic programming techniques without the need for external processing through transcription, neural features, or synthesis. We hope that our implementation provides a simple and directly accessible tool for the community. 

Our implementation is available online: https:// github.com/sildater/parangonar 

## Acknowledgments

This research acknowledges support by the Linz Institute of Technology Artificial Intelligence Lab and the by the European Research Council (ERC), under the European Union’s Horizon 2020 research and innovation programme, grant agreement No. 101019375 Whither Music?. 

## 5. REFERENCES



[1] T. Kwon, D. Jeong, and J. Nam, “Audio-to-score alignment of piano music using rnn-based automatic music transcription,” in Proceedings of the 14th Sound and Music Computing Conference (SMC), 2017. 





[2] S. Murgul, M. Reiser, M. Heizmann, and C. Seibert, “Fine-tuning midi-to-audio alignment using a neural network on piano roll and cqt representations,” arXiv preprint arXiv:2506.22237, 2025. 





[3] J. Zeitler, B. Maman, and M. Muller, “Robust and ac-¨ curate audio synchronization using raw features from transcription models.” in Proceedings of the International Society of Music Information Retrieval Conference (ISMIR), 2024, pp. 120–127. 





[4] F. Simonetta, S. Ntalampiras, and F. Avanzini, “Audioto-score alignment using deep automatic music transcription,” in 23rd International Workshop on Multimedia Signal Processing (MMSP), 2021. 





[5] S. D. Peter, P. Hu, and G. Widmer, “Pairing real-time piano transcription with symbol-level tracking for precise and robust score following,” in Proceedings of the Sound and Music Computing Conference (SMC), 2025. 





[6] S. Bock and G. Widmer, “Maximum filter vibrato sup-¨ pression for onset detection,” in Proceedings of the 16th International Conference on Digital Audio Effects (DAFx-13), Maynooth, Ireland, September 2013. 





[7] S. D. Peter, C. E. Cancino-Chacon, F. Foscarin, A. P.´ McLeod, F. Henkel, E. Karystinaios, and G. Widmer, “Automatic note-level score-to-performance alignments in the asap dataset,” Transactions of the International Society for Music Information Retrieval (TISMIR), 2023. 





[8] M. Muller, Y.¨ Ozer, M. Krause, T. Pr<sup>¨</sup> atzlich, and¨ J. Driedger, “Sync toolbox: A python package for efficient, robust, and accurate music synchronization,” Journal of Open Source Software, vol. 6, no. 64, p. 3434, 2021. 

