# PDF to md form
## Intro:
Helping people get CS field paper's main content.

## Usage:

### Preparation Requirement:
Install relative package

Device: Plz read README.md in [marker](https://github.com/VikParuchuri/marker)

Model: text limit larger than 32k

### Procedure:
In your terminal:
```shell
pip install marker-pdf
```

In your Github:
- create a new repo.
- create a README.md.
- create a README_CN.md. (for those who need translation other language needs revise the translation api prompt)
- creat a img folder as picture repo

In code:
- Result repo URL changed in [main.py](green_work/main.py)
- API: This frame's communication API connects to OpenAI's. Necessarily revise in [api_prompt.py](green_work/main_struct/api_prompt/api_prompt.py) & [api.py](green_work/translation/api.py). Key & URL are revised in [main.py](green_work/main.py)

In terminal:
```shell
marker /path/to/input/folder /temporary/md_file/restore/folder --workers 1 --max 10 --min_length 10000
```
- `/path/to/input/folder` is a folder with a few of PDF files.
- `/temporary/md_file/restore/folder` is a new folder (created by user) that store the result of recognition result.
- `--workers` default is 1 (recommend default).
- `--max` is the num of PDF files.
- `--min_length` is the minimum number of characters that need to be extracted from a pdf before it will be considered for processing.  If you're processing a lot of pdfs, I recommend setting this to avoid OCRing pdfs that are mostly images. (slows everything down)
- for more about paper's language settings, plz check [marker](https://github.com/VikParuchuri/marker)
