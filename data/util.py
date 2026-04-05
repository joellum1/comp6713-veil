def ensure_dirs():
    sentiment_datasets = os.path.join(BASE, "sentiment_datasets")
    raw_dir = os.path.join(sentiment_datasets, "raw")
    processed_dir = os.path.join(sentiment_datasets, "processed")

    fpb_dir = os.path.join(raw_dir, "FPB")
    fmb_dir = os.path.join(raw_dir, "FMB")

    os.makedirs(fpb_dir, exist_ok=True)
    os.makedirs(fmb_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    return {
        "sentiment_datasets": sentiment_datasets,
        "raw": raw_dir,
        "processed": processed_dir,
        "fpb": fpb_dir,
        "fmb": fmb_dir,
    }
