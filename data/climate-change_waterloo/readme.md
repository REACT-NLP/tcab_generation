## Climate-Change Waterloo Dataset

1. Download the `climate-change_waterloo` dataset:
	1. Download `climate_id.txt.00` file from [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5QCCUU) to this directory.
1. Preprocess the dataset:
	1. Hydrate the dataset using [twarc](https://twarc-project.readthedocs.io/en/latest/twarc2_en_us/).
	2. Randomly sample 43,943 instances, and save the data as `twitter_sentiment_data.csv` with columns: `sentiment`, `message`, `tweetid`.
   1. Run `python3 preprocess.py`. This generates a `train.csv`, `val.csv`, and `test.csv`.
