# Dataset access

The datasets are intentionally not redistributed in this package.

## ASAP Automated Essay Scoring dataset

Download from:

https://www.kaggle.com/competitions/asap-aes

Users must comply with the access and use conditions shown on Kaggle.

## Operating System short-answer dataset

Download from:

https://github.com/RAMESHDADI/OS-DATA_1-SET-FOR-AES

Users should consult the repository for its current files, documentation, and
applicable terms.

After downloading a dataset, create a task-specific CSV containing `ID`,
`essay`, and `score`. Save the sample task as `data/ASAP1.csv` or
`data/OS3.csv`, or set `AES_DATASET_CSV` to the file's location. The package
does not contain student-response text.
