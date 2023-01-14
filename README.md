# ID2223

This repository contains solutions for the ID2223 Scalable Machine Learning and Deep Learning course at KTH.

# Project
### Description
This project is an advanced adaptation from the air quality prediction project found in [Hopsworks Air Quality Tutorial](https://github.com/logicalclocks/hopsworks-tutorials/tree/master/advanced_tutorials/air_quality "Hopsworks Air Quality Tutorial")

Similarly to the original Air Quality Project this project perdicts air quailty based on various weather metrics. This project however, aims to predict the air quality over the next 7 days (including today) across the country of Poland. The Goal is to be able to visualize how air quality metrics shift country-wide across multiple days in the future.

### Link to public UI
A hosted public UI of the project can be found [Here](https://erno98-id2223-air-qualitystreamlit-app-p8sjf5.streamlit.app/ "PM10 Predictions for Poland")
<br>**Note:** This project is not optimised for all Web Browsers e.g Safari, our testing was done on Google Chrome version: 108.0.5359.124.

https://erno98-id2223-air-qualitystreamlit-app-p8sjf5.streamlit.app/

### Prerequisites
In order to run this project you will need to connect to a Hopswoks account which will provide you with a Hopsworks API Key. You will then export to your Hopsworks API Key as an environment variable in the project. 

You will also be required to install existing Python libraries used in the project such as for example Streamlit.

### Notes
If desired you can choose to export the collected weather data used for predictions as a CSV file by setting ```save_to_csv``` parameter to true when ```get_weather_data``` is called.

### Structure
Contained within the air_quality directory with following structure:
```
air_quality   
│   1_backfill_feature_groups.ipynb
│   2_queries_and_merging.ipynb
│   3_training.ipynb
│   AirParticle_Forest.pkl
│   Gradient Duster.pkl
│   GradientBoostingRegressor.pkl
│   Gradient_Duster.pkl
│   PM10Lasso.pkl
│   README.md
│   RandomForestRegressor.pkl
│   requirements.txt   ← requirements for the whole project 
│   streamlit_app.py
│   testing.ipynb
│
└───.ipynb_checkpoints                              ← Checkpoints from the Jupyter notebooks
│   │   1_backfill_feature_groups-checkpoint.ipynb
│   │   2_queries_and_merging-checkpoint.ipynb
│   │   3_training-checkpoint.ipynb
│   │   testing-checkpoint.ipynb
│
└───__pycache__
│   │   functions.cpython-37.pyc
│   │   functions.cpython-38.pyc
│
└───data_poland
│   │   meta.xlsx
│   │   visualize.ipynb
│   │   
│   └───.ipynb_checkpoints 
│   │   │   Untitled-checkpoint.ipynb
│   │   │   collect_metadata-checkpoint.ipynb
│   │   │   visualize-checkpoint.ipynb
│   │   │   visualize_test-checkpoint.ipynb
│   │   
│   └───air_quality
│   │   │   air_quality_merged.csv
│   │   │   biala_podlaska.csv
│   │   │   bialystok.csv
│   │   │   bielsko_biala.csv
│   │   │   bydgoszcz.csv
│   │   │   gdansk.csv
│   │   │   gorzow.csv
│   │   │   kalisz.csv
│   │   │   katowice.csv
│   │   │   koszalin.csv
│   │   │   krakow.csv
│   │   │   lodz.csv
│   │   │   lomza.csv
│   │   │   lublin.csv
│   │   │   merge_air_quality.ipynb
│   │   │   poznan.csv
│   │   │   radom.csv
│   │   │   rzeszow.csv
│   │   │   suwalki.csv
│   │   │   szczecin.csv
│   │   │   szczecinek.csv
│   │   │   warszawa.csv
│   │   │   wroclaw.csv   
│   │   └───.ipynb_checkpoints
│   │       │  merge_air_quality-checkpoint.ipynb
│   │   
│   └───dummy_data
│   │   │   pred.xlsx
│   │   │   true.xlsx
│   │   
│   └───weather
│       │   biala_podlaska.csv
│       │   bielsko_biala.csv
│       │   gdansk.csv
│       │   katowice.csv
│       │   krakow.csv
│       │   merge_weather.ipynb
│       │   poznan.csv
│       │   warszawa.csv
│       │   weather_merged.csv
│       │   wroclaw.csv
│       └───.ipynb_checkpoints
│           │  merge_weather-checkpoint.ipynb
│
└───functions
│   │   functions.py
│   │   get_weather_data.py
│   └───__pycache__
│       │   functions.cpython-37.pyc
│       │   functions.cpython-38.pyc
│       │   get_weather_data.cpython-37.pyc
│
└───images
│   │   1.png
│   │   2.png
│   │   api_keys_env_file.png
│
└───weather_files
│   │   2023-01-13.csv
```

# Lab 2
Contained within the `swedish_fine_tune_whisper` directory with following structure: 
```
swedish_fine_tune_whisper
│   env.yml               ← requirements for the conda environment for pipelines
│   feature_pipeline.py   ← code for extracting the features from common voice dataset
│   training_pipeline.py  ← code for training the whisper model with extracted dataset
│   training_config.json  ← config file for training, used in training_pipeline
│   utils.py              ← helper functions
│   huggingface_token.txt ← token for huggingface, here it's of course empty
```
For this project, we stored the data on google drive, which we then mounted into a shared collab to perform the training. This made the work flow pretty efficient, and the size of the data was not big enough to cause problems on our drive spaces.

To improve the model performance with **model-centric approach** we can:
- tune the training parameters:
  - by increasing ```num_train_epochs``` we make the model learn for longer, which fits it better to the data. However, it's really slow, and we need to watch out not to overtrain. Please note, that setting up the ```max_steps``` will override this variable.
  - tweaking the ```learning_rate``` - higher learning rate may cause quicker optimization, however if too big, we may overshoot potential maxima.
  - changing the ```per_device_train_batch_size``` and ```gradient_accumulation_steps``` as they imply how many samples we feed to the model for each step in gradient calculation, so with more data the gradient change will be slower. These two parameters are highly connected.
  - ```fp16``` greatly improves the speed of traning (available only on GPU)
- We can use a bigger model of Whisper. We're using the small variation (244M params), which was pretrained only on English. There's also medium and large, to have a more complex models that could become necessary for difficult languages. 
- Use different model - some Wav2Vec variations have about 6% WER on the [hugging face benchmark](https://paperswithcode.com/sota/speech-recognition-on-common-voice-swedish)

However, when it comes to **data-centric approach**, we may:
- Include more data - this however is problematic with languages such as Swedish, as there are not much reliable alternatives to Common Voice. We managed to identify:
  - [Europarl (European Parliament Proceedings Parallel Corpus)](https://paperswithcode.com/dataset/europarl)
  - [CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus](https://paperswithcode.com/dataset/covost)
  
  However, we did not have time to incorporate it into our pipeline.

The HuggingFace GradIo can be found here: https://huggingface.co/spaces/Danker/Whisper

# Lab 1
Contained within the `serverless-ml-intro` directory with following structure: 
```
serverless-ml-intro
│   requirements.txt   ← requirements for the whole project 
│
└───iris                              ← project for the Iris dataset
│   │   iris-batch-inference-pipeline.py ← prediction pipeline
│   │   iris-feature-pipeline-daily.py   ← daily prediction
│   │   iris-feature-pipeline.py         ← creating feature group
│   │   iris-training-pipeline.py        ← training pipeline
│   └───huggingface-spaces-iris            ← huggingface app for predictions
│       │   README.md
│       │   app.py
│       │   requirements.txt
│   └───huggingface-spaces-iris-monitor    ← huggingface app for monitoring predictions
│       │   README.md
│       │   app.py
│       │   requirements.txt
│   
└───titanic                             ← project for the Titanic dataset
│   │   titanic-batch-inference-pipeline.py ← prediction pipeline
│   │   titanic-feature-pipeline-daily.py   ← daily prediction
│   │   titanic-feature-pipeline.py         ← creating feature group
│   │   titanic-training-pipeline.py        ← training pipeline
│   │   preprocess_titanic.ipynb            ← notebook for processing the titanic data
│   └───huggingface-spaces-titanic            ← huggingface app for predictions
│       │   README.md
│       │   app.py
│       │   requirements.txt
│   └───huggingface-spaces-titanic-monitor    ← huggingface app for monitoring predictions
│       │   README.md
│       │   app.py
│       │   requirements.txt
```
Apart from the aforementioned, there are pictures, files, and datasets generated by the described scripts.

The huggingface apps are available under following links:
- Iris predictions: https://huggingface.co/spaces/Danker/ID2223_iris
- Iris monitoring (daily): https://huggingface.co/spaces/Danker/iris_daily
- **Titanic** predictions: https://huggingface.co/spaces/Danker/ID2223_Titanic
- **Titanic** monitoring (daily): https://huggingface.co/spaces/Danker/titanic_daily

## Titanic data processing
The raw titanic data was processed to fit out prediction model (in our case, **Random Forest Classifier** (RFC) from sklearn). Following changes has been done:
- attribute Embarked was made numerical (S→0, C→1, Q→2)
- attribute Sex was made numerical (female→1, male→0)
- non-predictive attributes were dropped: PassengerId, Name, Ticket, Cabin
- rows containing empty (NaN) values were dropped from the dataset
- the whole dataset was cast as containing float values (due to some issues with hopsworks)

The RFC was trained using 0.2 train-test-split fraction, achieving accuracy of 79% on the test set.
