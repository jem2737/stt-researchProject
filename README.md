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

### [Scripts](/documentation/Speaker_class_scripts)

## Audio requirements 
* .wav
* mono if possible
* 16-bit PCM
* 16 kHz or 44.1 kHz

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