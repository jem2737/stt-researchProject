# Think[box] Micro-Grant
## Intended File Structure
## Stress class Defs
* Normal
  * Steady speaking rate
  * Normal loudness
  * Regular breathing
  * Clear articulation
  * No strong tension, panic, or strain in the voice
* Stressed
  * Somewhat faster or uneven speaking rate
  * Raised vocal intensity or tighter tone
  * Mild shakiness, breathiness, or tension
  * More effort in the voice
  * Sounds pressured, frustrated, or strained, but still somewhat controlled
* Distressed
  * strong urgency or panic in the voice
  * irregular pacing or broken speech
  * yelling, gasping, strained shouting, or clipped words
  * unstable loudness
  * heavy breathing or vocal struggle
  * clear sense that something is wrong right now

### [Scripts](/documentation/Speaker_class_scripts/firefighter_voice_scripts.pdf)

## Audio requirements 
* .wav
* mono if possible
* 16-bit PCM
* 16 kHz or 44.1 kHz

# Data

## Custom Audio
- I recorded custom audio data with 5 volunteers reading short scripts designed to reflect the three target classifications: normal, stressed, and distressed.
- Each script was approximately 6 seconds long.
- Each speaker recorded 10 clips per class.
- This resulted in a total of 150 custom audio clips.

## Supplementary Dataset Audio: Rosie-Lab BERSt
- A dataset of 150 clips is not large enough for robust model training, so I am supplementing my custom recordings with audio from the Rosie-Lab BERSt dataset.
- BERSt contains many metadata fields, but the fields most relevant to this project are:
  - `phone_position`
  - `shout_level`
  - `affect`

### Phone Position Filtering
To make the audio as consistent as possible with the intended use case, I only use BERSt clips with phone positions that simulate a normal phone conversation. The phone position categories used are:
- `Hold your phone next to your face with the mic facing your mouth as you would in a phone conversation`
- `Place phone 1–2 meters away face up on any surface`

### Class Mapping
The BERSt dataset does not directly contain the classes normal, stressed, and distressed, so I define those classes using the available metadata:

- **Normal**
  - Affect: `neutral`
  - Shout level: `no shout`

- **Stressed**
  - Affect: `anger`, `fear`, `sadness`, `disgust`, `surprise`
  - Shout level: `no shout`

- **Distressed**
  - Affect: `anger`, `fear`, `sadness`, `disgust`, `surprise`
  - Shout level: `shout`

### BERSt Sampling Plan
- To expand the training set, I randomly select 25 BERSt clips per class that match the required metadata.
- These clips are used to supplement the custom audio recordings.
- The goal of adding BERSt data is to increase dataset size and improve diversity.
- This is especially important because my custom dataset currently includes recordings from 1 female speaker and 4 male speakers.
  
# project steps:
* clearly define the project goal
* decide exactly what you are benchmarking
* choose the task you care about
* choose the models you want to compare
* set up the Jetson software environment
* confirm each model can run on the Jetson
* collect and organize your audio dataset
* make sure all files are in a consistent format
* label the fi*les clearly
* create a repeatable audio loading pipeline
* inspect the dataset for bad recordings, silence, clipping, or noise issues
* decide how the audio will be fed into each model
* preprocess the audio into the format each model expects
* build a script that runs inference on one file
* build a script that runs inference on the full dataset
* record each model’s predicted output
* measure inference time per clip
* measure total runtime
* measure memory usage
* measure CPU usage
* measure GPU usage if applicable
* measure power usage if you are including efficiency
* record model size
* record preprocessing time
* separate preprocessing time from inference time
* evaluate accuracy or classification performance
* create a confusion matrix if it is a classification task
* compare results across all models
* compare results across different audio lengths if relevant
* compare results across different recording conditions if relevant
* document all software versions and settings
* document Jetson hardware configuration
* make sure the benchmarking procedure is repeatable
* save all results in a structured format
* make tables and plots for the benchmark results
* decide which model gives the best tradeoff between speed, size, and accuracy
* write up your conclusions for Jetson deployment