# sequential_recommender_system_sagemaker_with_custom_container
- This project is for testing sequential recommender system and sagemaker for deployment
- The research process can be seen here: https://joyous-snout-4cc.notion.site/Web-Fiction-Recommender-System-7fa2d333f5c84378ae8bea53b665b9f7

## Directory Structure
<img width="942" alt="Screenshot 2022-12-30 at 2 09 11 PM" src="https://user-images.githubusercontent.com/43153661/210163319-2983750e-0fcf-4eb7-bf9a-c3b85b76e0b0.png">
- [input]
  - movies.csv -> information of movies(title, genre)
  - ratings.csv -> rating data of each user
- [main] : each Dockerfile is used for build container for each step of sagemaker
  - embbedding.py -> get embedding and calculate similarity
  - preprocess.py -> preprocess data for training
  - train.py -> train recommendation mode
  - inference.py -> use for inference (Sagemaker endpoint)

```
.
├── Dockerfile_embedd
├── Dockerfile_inference
├── Dockerfile_preprocess
├── Dockerfile_train
├── README.md
├── embedding.py
├── inference.py
├── input
│   ├── movies.csv
│   └── ratings.csv
├── preprocessing.py
├── src
│   ├── __init__.py
│   ├── components
│   │   ├── __init__.py
│   │   ├── chatie_ml.py
│   │   ├── embed
│   │   │   ├── __init__.py
│   │   │   ├── embed.py
│   │   │   ├── image_embed.py
│   │   │   └── text_embed.py
│   │   ├── inference
│   │   │   ├── __init__.py
│   │   │   └── inference.py
│   │   ├── preprocess
│   │   │   ├── __init__.py
│   │   │   └── preprocess.py
│   │   ├── similarity
│   │   │   ├── __init__.py
│   │   │   └── similarity.py
│   │   └── train
│   │       ├── __init__.py
│   │       └── train.py
│   └── module
│       ├── __init__.py
│       ├── db
│       │   ├── __init__.py
│       │   ├── bigquery.py
│       │   ├── db.py
│       │   ├── firebase.py
│       │   ├── mysql.py
│       │   └── s3.py
│       ├── metrics.py
│       ├── resource_manager
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── embed
│       │       ├── __init__.py
│       │       └── base.py
│       └── utils
│           ├── __init__.py
│           ├── common.py
│           ├── data.py
│           ├── gpu.py
│           ├── metrics.py
│           ├── structure.py
│           └── text.py
└── train.py
```
