## Wikipedia (Talk Pages) Dataset

1. Download the `wikipedia` dataset:
	1. Go to [https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973](https://figshare.com/articles/Wikipedia_Talk_Labels_Toxicity/4563973).
	1. Download the following files to this directory:
		* `toxicity_annotations.tsv`.
		* `toxicity_annotated_comments.tsv`.
1. Preprocess the dataset:
   1. Run `python3 preprocess.py`. This should generate a `train.csv`, `val.csv`, and `test.csv`.
