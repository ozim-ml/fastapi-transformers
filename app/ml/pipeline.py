import time
import numpy as np
import pandas as pd
from transformers import pipeline
from nltk.tokenize import PunktSentenceTokenizer
from threading import Lock

punkt_tokenizer = PunktSentenceTokenizer()
clf_lock = Lock()


class OneShotModel:
    """
    A wrapper for running zero-shot text classification across multiple tasks.

    Processes input text, splits it into chunks, applies a zero-shot classification pipeline, 
    and aggregates results for each task defined in the provided model configuration dictionary.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the text to classify.
    text_col : str
        Name of the column in the DataFrame that holds the text.
    model_dict : list
        A list of task configurations, each containing labels, chunk_size, etc.
    clf_pipe : transformers.Pipeline, optional
        A zero-shot classification pipeline. If not provided, a default BART MNLI model is created.
    display_time : bool, default True
        Whether to print processing time for each task.

    Attributes
    ----------
    labels : pandas.DataFrame
        The predicted labels for each task.
    scores : pandas.DataFrame
        The associated confidence scores for each task.
    _chunk_cache : dict
        Internal cache storing chunk-level predictions per task.
    """

    def __init__(self, df, text_col, model_dict, clf_pipe=None, display_time=True):
        self.df = df.copy()
        self.text_col = text_col
        self.model_dict = model_dict
        self.display_time = display_time
        self.clf_pipe = clf_pipe or pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        self.labels = pd.DataFrame(index=df.index)
        self.scores = pd.DataFrame(index=df.index)
        self._chunk_cache = {}

        self._run_all()

    def _run_all(self):
        """
        Execute zero-shot classification for all configured tasks.

        Iterates over each task, applies the pipeline, and stores predictions and scores internally.
        """
        for i, model_cfg in enumerate(self.model_dict):
            labels = model_cfg["labels"]
            chunk_size = model_cfg.get("chunk_size", 2)

            label_df, score_df, cache = self._zero_shot_pipe(i, labels, chunk_size=chunk_size)
            self.labels[f"label_{i}"] = label_df
            self.scores[f"score_{i}"] = score_df
            self._chunk_cache[i] = cache

    def _zero_shot_pipe(self, task_idx, labels, chunk_size=2):
        """
        Run zero-shot classification for a single task.

        Splits the input text into sentence chunks, applies the zero-shot classifier, 
        selects a label based on adaptive scoring, and stores chunk-level results.

        Parameters
        ----------
        task_idx : int
            The index of the current task in the model dictionary.
        labels : list
            List of candidate labels for this task.
        chunk_size : int, default 2
            The number of sentences to group into each chunk.

        Returns
        -------
        pandas.Series
            Predicted labels for each row in the DataFrame.
        pandas.Series
            Confidence scores associated with the selected labels.
        dict
            Cache dictionary containing chunk-level results for this task.
        """
        texts = self.df[self.text_col].fillna("").astype(str)
        empty_mask = texts.str.strip() == ""
        non_empty_texts = texts[~empty_mask].tolist()

        start_time = time.time()
        results, cache = [], {}

        for df_idx, text in zip(self.df.index[~empty_mask], non_empty_texts):
            sentences = punkt_tokenizer.tokenize(text)
            chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

            with clf_lock:
                chunk_results = self.clf_pipe(
                    chunks,
                    candidate_labels=labels,
                    multi_label=False,
                    hypothesis_template="This message is about {}.",
                    batch_size=16,
                    truncation=True,
                )

            best_chunk = max(chunk_results, key=lambda r: r["scores"][0])
            best_label = best_chunk["labels"][0]
            best_score = best_chunk["scores"][0]

            mean_score = np.mean(best_chunk["scores"])
            std_score = np.std(best_chunk["scores"])
            adaptive_threshold = mean_score + 0.5 * std_score

            if best_score < adaptive_threshold:
                best_label = "uncertain"
                best_score = np.nan

            results.append({"label": best_label, "score": best_score})
            cache[df_idx] = {
                "text": text,
                "chunks": chunks,
                "results": chunk_results,
                "selected_label": best_label,
                "selected_score": best_score,
            }

        end_time = time.time()
        if self.display_time:
            print(f"[Task {task_idx}] completed in {end_time - start_time:.2f}s")

        def yield_predictions():
            iter_results = iter(results)
            for is_empty in empty_mask:
                if is_empty:
                    yield None, np.nan
                else:
                    r = next(iter_results)
                    yield r["label"], r["score"]

        pred_labels, pred_scores = zip(*yield_predictions())
        label_series = pd.Series(pred_labels, index=self.df.index)
        score_series = pd.Series(pred_scores, index=self.df.index)

        return label_series, score_series, cache