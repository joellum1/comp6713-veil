# COMP6713 - VEIL - Financial News Sentiment Analysis

Sentiment analysis classification on financial news.

---

## Datasets

### Sentiment Analysis Datasets

| Dataset | Source |
|---|---|
| Financial PhraseBank | [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) |
| FinMarBa Dataset | [Hugging Face](https://huggingface.co/datasets/baptle/financial_headlines_market_based) |

### News Summarisation Datasets

| Dataset | Source |
|---|---|
| CNN-DailyMail News Text Summarisation | [Kaggle](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/data) |
| News Summary | [Kaggle](https://www.kaggle.com/datasets/sunnysai12345/news-summary) |
 
See `data/README.md` for full download and placement instructions.

---

## Project Execution

### Run `notebooks/langchain_pipeline_demo.ipynb` from scratch

The LangChain demo notebook uses BART + FinBERT + L&M lexicon pipeline
implemented in `src/pipeline/`.

Use the steps below exactly (recommended for a clean first run):

```bash
# 1) Create and activate a clean Python 3.11 environment
conda create -n 6731 python=3.11 -y
conda activate 6731

# 2) Enter project root
cd /path/to/comp6713-veil

# 3) Install dedicated notebook/pipeline dependencies
#    (use this file, not requirements.txt)
pip install -r requirements-pipeline.txt

# 4) Register Jupyter kernel for this env
python -m ipykernel install --user --name 6731 --display-name "Python (6731)"

# 5) (optional) Drop in pre-trained checkpoints — see "Pre-trained checkpoints" below

# 6) Launch notebook
jupyter notebook notebooks/langchain_pipeline_demo.ipynb
```

In Jupyter:
1. Switch kernel to **Python (6731)**.
2. Restart kernel once.
3. Run all cells from top to bottom.

#### Pre-trained checkpoints (optional)

If you don't want to fine-tune locally (BART training on CNN/DailyMail takes
hours), download the team's checkpoints from Google Drive and place them under
`results/`. The pipeline will pick them up automatically:

| Model | Drive folder | Local destination |
|---|---|---|
| BART (News Summary) | [bart-final-ns](https://drive.google.com/drive/folders/1p5FuQAvQoguGp_hCBpqYiSF5_ofr9K9W?usp=sharing) | `results/bart-final-ns/` |
| FinBERT | [finbert_model](https://drive.google.com/drive/folders/1udKH-F7yfQeDBfP06MpvHBJwgCfDmwS7?usp=sharing) | `results/finbert_model/` |

```bash
mkdir -p results/bart-final-ns results/finbert_model
# Then download every file inside each Drive folder into the matching
# local directory (config.json, model.safetensors, tokenizer.json, ...).
```

Without these, the pipeline auto-downloads `facebook/bart-large-cnn` (~1.6 GB)
and `ProsusAI/finbert` (~400 MB) from the Hugging Face Hub on first run.

If you see dependency errors, verify versions:

```bash
python -c "import numpy, torch, transformers; print(numpy.__version__, torch.__version__, transformers.__version__)"
```

Expected compatible range:
- `numpy < 2`
- `transformers >= 4.49, < 5`
- `torch >= 2.2`

---

## Testing CMD Interface

### Preliminaries
Extract the model zip files provided in the `./src/models` directory. If you are testing the summary dataset, upload the `.txt` file to be tested in the `./src/test/testing_data` directory.

### Running the Test
After completing the preliminaries, run the following command:
```bash
python3 ./src/test/test_models.py
```
