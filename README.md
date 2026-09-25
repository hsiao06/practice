Based on https://docs.pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html

Adapted to typical ASR setting: 25ms window length, 10ms window shift, Mel filter bank features

* fe_epi.py
  * load wav/epi.wav
  * load audio, resample to 8kHz, extract spectrogram and mel-filterbank features
  * make plots
* fe_speech_and_music.py
  * load wav/speech-and-music.wav
  * load audio, resample to 16kHz, extract spectrogram and mel-filterbank features
  * average the log Mel filter bank features from the two channels
* run_speech_and_music_kmeans.py
  * demonstrate the we can cluster the audio into speech and music portions
* run_speech_and_music_mlp.py
  * train a simple MLP to predict the speech and music, assuming the k-means as labels
* run_speech_and_music_cnn.py
  * train a CNN with average pooling to enable smoothed predictions
