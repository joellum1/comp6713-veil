# Contribution

## Gala (zID - z5242812) (Part C - LangChain integration)

End-to-end orchestration layer that wires the team's models into a single
schema-validated reliability report.

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


## Raymond (zID - z5479272)
1. Led the report integration for the sentiment analysis component, turning technical implementation details from multiple teammates into a coherent modelling and evaluation narrative.
2. Wrote and refined the report sections for Problem Definition, Dataset Selection, and the FinBERT sentiment modelling and quantitative evaluation components.
3. Helped debug the sentiment data-processing workflow by identifying the earlier duplicate concatenation / double-counting issue in the processed dataset pipeline and confirming the corrected final setup with teammates.
4. Synthesised the rationale behind the three-stage sentiment pipeline (Loughran–McDonald baseline, TF-IDF + SVM baseline, and fine-tuned FinBERT) to ensure the report clearly explained model progression and justification.
5. Interpreted quantitative results from the final FinBERT run, including accuracy, weighted F1, validation curves, and confusion matrix behaviour, and translated them into report-ready analysis.
6. Identified cross-section inconsistencies and clarified final modelling decisions with teammates, including dataset usage, class imbalance treatment, weighted loss implementation, and the role of lexicon-based components.
7. Helped build the presentation for the sentiment analysis section by converting the report content into slide material for a technical audience and maintained aesthetic design.

