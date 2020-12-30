# Equal-Accuracy-Ratio
---
## Multidialect Data
- African American English (CORAAL) <https://oraal.uoregon.edu/coraal>
- Standard American English (LibriSpeech) <http://www.openslr.org/12>
- Hispanic English (LDC2014S05) <https://www.ldc.upenn.edu>
- British English (WSJCAM0 Cambridge Read News) <https://www.ldc.upenn.edu>
- Afrikaans English (AST Afrikaans English) <https://vlo.clarin.eu/record/https_58__47__47_hdl.handle.net_47_20.500.12185_47_411_64_format_61_cmdi?2>
- Xhosa English (AST Black English) <https://repo.sadilar.org/handle/20.500.12185/433>
- Inidian English  (Voxforge) <http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/8kHz_16bit>
---
## Preprocessing
- Preprocessing script is located in scripts/preprocessing. Change the data directory in the *generate_transcription_meta.ipynb* script under each dialect folder and execute cells to get metadata for each dialect.
- Execute cells in *extract_feature_stft.ipynb* to extract STFT features and save as *.npy* format.
- Execute cells in *combine_transcriptions.ipynb* to merge the metadata of all dialect.
---
## Experiment
### Dialect Experiment
`python -u train_fair.py --config config/deepspeech_dialect.yml --sample-size <sample-size> --seed <seed> --ita <equal-acc-ratio-weight> --save-dir <save-dir>`

### Coraal Experiment
```
for attr in work age edu gender
do
    for i in 0 0.001 0.01 0.1 1 10
    do
        python train_fair.py --config config/coraal_rnnctc/coraal_${attr}.yml` --sample-size <sample-size> --ita ${i}
		--save-dir checkpoints/basectc_coraal_${attr}_ita${i}
    done
done
```
