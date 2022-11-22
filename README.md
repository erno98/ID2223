# ID2223

This repository contains solutions for the ID2223 Scalable Machine Learning and Deep Learning course at KTH.

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
│   │   titanic-training-pipeline.py.       ← training pipeline
│   └───huggingface-spaces-titanic            ← huggingface app for predictions
│       │   README.md
│       │   app.py
│       │   requirements.txt
│   └───huggingface-spaces-titanic-monitor    ← huggingface app for monitoring predictions
│       │   README.md
│       │   app.py
│       │   requirements.txt
```
The huggingface apps are available under following links:
- Iris predictions: https://huggingface.co/spaces/Danker/ID2223_iris
- Iris monitoring (daily): https://huggingface.co/spaces/Danker/iris_daily
- Titanic predictions: https://huggingface.co/spaces/Danker/ID2223_Titanic
- Titanic monitoring (daily): https://huggingface.co/spaces/Danker/titanic_daily
