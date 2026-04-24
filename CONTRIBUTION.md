# Contribution

## Gala (zID - z5242812)

1. Designed and implemented the end-to-end LangChain reliability pipeline.
2. Refactored the two summarisation notebooks (`notebooks/summarisation.ipynb` and `notebooks/summarisation_ns.ipynb`).
3. Optimised BART fine-tuning hyperparameters across CNN/DailyMail and News Summary, ran the training, and produced the quantitative ROUGE results.
4. Contributed to the report (wrote the *System Integration with LangChain* subsection of the Modelling chapter).

## Joel (zID - z5481782)

1. Developed data processing pipeline (retrieval, cleaning, preprocessing) for summarisation.
2. Implemented rule-based baseline for summarisation.
3. Researched suitable pre-trained models for summarisaiton, and worked on fine-tuning BART.
4. Qualitatively evaluated fine-tuned BART.
5. Contributed to report (Part C and D for summarisation model)

## Omar (zID - z5383441)
1. Developed data processing pipeline (retrieval, cleaning, preprocessing) for sentiment analysis.
2. Implemented baseline rule-based and statistical sentiment models.
3. Researched hybrid FinBERT-lexicon integration for sentiment analysis.
4. Developed command-line testing interface for model evaluation.
5. Wrote report (Part C section) for statistical model for sentiment analysis.

## Alex (zID - z5593503)
1. Researched hybrid FinBERT-lexicon integration for sentiment analysis.
2. Implemented the sentiment analysis pipeline using FinBERT, including model loading, training, and evaluation logic.
3. Developed a custom WeightedTrainer to apply Weighted Cross-Entropy Loss, addressing the class imbalance between negative and majority samples.
4. Configured fine-tuning hyperparameters (learning rate, weight decay, and warmup) to stabilize training on a small financial dataset.
5. Built a bias-scoring module using the sentiment_list.csv built using Loughran-McDonald (L&M) dictionary
6. Created evaluation definitions to output confusion matrices and learning curves for performance tracking.
7. Performed Qualitative Evaluation of model predictions, identifying and categorizing common misclassification patterns.
8. Contributed to the report sections regarding model design, training result, and error analysis.
