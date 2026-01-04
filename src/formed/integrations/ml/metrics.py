"""Metrics for evaluating machine learning models.

This module provides a comprehensive set of metrics for evaluating
classification, regression, and ranking models. All metrics follow
a common interface with reset, update, and compute methods.

Key Components:
    Base Classes:

    - `BaseMetric`: Abstract base for all metrics
    - `BinaryClassificationMetric`: Base for binary classification metrics
    - `MulticlassClassificationMetric`: Base for multiclass metrics
    - `MultilabelClassificationMetric`: Base for multilabel metrics
    - `RegressionMetric`: Base for regression metrics
    - `RankingMetric`: Base for ranking metrics

    Classification Metrics:

    - `BinaryAccuracy`: Binary classification accuracy
    - `BinaryFBeta`: Binary F-beta score (precision, recall, F1)
    - `BinaryROCAUC`: Binary ROC AUC (requires scores)
    - `BinaryPRAUC`: Binary PR AUC (Precision-Recall curve, requires scores)
    - `MulticlassAccuracy`: Multiclass accuracy (micro/macro)
    - `MulticlassFBeta`: Multiclass F-beta (micro/macro)
    - `MultilabelAccuracy`: Multilabel accuracy
    - `MultilabelFBeta`: Multilabel F-beta

    Regression Metrics:

    - `MeanAbsoluteError`: MAE metric
    - `MeanSquaredError`: MSE metric

    Ranking Metrics:

    - `MeanAveragePrecision`: MAP metric
    - `NDCG`: Normalized Discounted Cumulative Gain

    Utility Metrics:

    - `Average`: Simple averaging metric
    - `EmptyMetric`: No-op metric

Features:
    - Stateful metrics with accumulation across batches
    - Support for micro and macro averaging
    - Flexible label types (`int`, `str`, `bool`)
    - Registrable for configuration-based instantiation

Examples:
    >>> from formed.integrations.ml.metrics import MulticlassAccuracy, ClassificationInput
    >>>
    >>> # Create metric
    >>> metric = MulticlassAccuracy(average="macro")
    >>>
    >>> # Update with batch
    >>> inputs = ClassificationInput(
    ...     predictions=[0, 1, 2, 1],
    ...     targets=[0, 1, 1, 1]
    ... )
    >>> metric.update(inputs)
    >>>
    >>> # Compute final metrics
    >>> results = metric.compute()
    >>> print(results)  # {"accuracy": 0.75}
    >>>
    >>> # Reset for next evaluation
    >>> metric.reset()

"""

import abc
import dataclasses
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar

from colt import Registrable

from .types import BinaryLabelT, LabelT

_T = TypeVar("_T")


class BaseMetric(Registrable, Generic[_T], abc.ABC):
    """Abstract base class for all metrics.

    Metrics are stateful objects that accumulate predictions and targets
    across multiple batches, then compute aggregate statistics.

    Type Parameters:
        _T: Type of input data for this metric.

    Examples:
        >>> @BaseMetric.register("my_metric")
        ... class MyMetric(BaseMetric[MyInputType]):
        ...     def reset(self):
        ...         self._state = 0
        ...     def update(self, inputs):
        ...         self._state += process(inputs)
        ...     def compute(self):
        ...         return {"metric": self._state}

    """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new evaluation.

        This should clear all accumulated statistics.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, inputs: _T) -> None:
        """Update internal state with a batch of predictions.

        Args:
            inputs: Batch of predictions and targets.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self) -> dict[str, float]:
        """Compute metrics from accumulated state.

        Returns:
            Dictionary mapping metric names to values.

        """
        raise NotImplementedError()

    def __call__(self, inputs: _T) -> dict[str, float]:
        """Update and compute in one call.

        Args:
            inputs: Batch of predictions and targets.

        Returns:
            Dictionary of computed metrics.

        """
        self.update(inputs)
        return self.compute()


@BaseMetric.register("empty")
class EmptyMetric(BaseMetric[Any]):
    """No-op metric that does nothing.

    This metric can be used as a placeholder when no evaluation is needed.

    """

    def reset(self) -> None:
        pass

    def update(self, inputs: Any) -> None:
        pass

    def compute(self) -> dict[str, float]:
        return {}


@BaseMetric.register("average")
class Average(BaseMetric[Sequence[float]]):
    """Simple averaging metric for numeric values.

    Computes the mean of all values seen across batches.

    Args:
        name: Name for the metric in output dictionary.

    Examples:
        >>> metric = Average(name="loss")
        >>> metric.update([1.0, 2.0, 3.0])
        >>> metric.update([4.0, 5.0])
        >>> metric.compute()  # {"loss": 3.0}

    """

    def __init__(self, name: str = "average") -> None:
        self._name = name
        self._total = 0.0
        self._count = 0

    def reset(self) -> None:
        self._total = 0.0
        self._count = 0

    def update(self, inputs: Sequence[float]) -> None:
        self._total += sum(inputs)
        self._count += len(inputs)

    def compute(self) -> dict[str, float]:
        return {self._name: self._total / self._count if self._count > 0 else 0.0}


@dataclasses.dataclass
class ClassificationInput(Generic[_T]):
    """Input data for classification metrics.

    Attributes:
        predictions: Sequence of predicted labels.
        targets: Sequence of ground truth labels.

    """

    predictions: Sequence[_T]
    targets: Sequence[_T]


@dataclasses.dataclass
class BinaryClassificationInput(Generic[BinaryLabelT]):
    """Input data for binary classification metrics that require probability scores.

    Attributes:
        predictions: Sequence of predicted labels.
        scores: Sequence of prediction scores (probabilities for positive class).
        targets: Sequence of ground truth labels.

    """

    predictions: Sequence[BinaryLabelT]
    targets: Sequence[BinaryLabelT]
    scores: Sequence[float] | None = None


class BinaryClassificationMetric(BaseMetric[BinaryClassificationInput[BinaryLabelT]], Generic[BinaryLabelT]):
    """Base class for binary classification metrics.

    Binary classification metrics work with two classes (`0 and `1``, or `True`/`False`).

    Type Parameters:
        BinaryLabelT: Type of labels (`int`, `bool`, etc.).

    """

    Input: type[BinaryClassificationInput[BinaryLabelT]] = BinaryClassificationInput


@BaseMetric.register("binary_accuracy")
@BinaryClassificationMetric.register("accuracy")
class BinaryAccuracy(BinaryClassificationMetric[BinaryLabelT], Generic[BinaryLabelT]):
    """Binary classification accuracy metric.

    Computes the fraction of correct predictions.

    Examples:
        >>> metric = BinaryAccuracy()
        >>> inputs = ClassificationInput(
        ...     predictions=[1, 0, 1, 1],
        ...     targets=[1, 0, 0, 1]
        ... )
        >>> metric.update(inputs)
        >>> metric.compute()  # {"accuracy": 0.75}

    """

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, inputs: BinaryClassificationInput[BinaryLabelT]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred, target in zip(predictions, targets):
            if pred == target:
                self._correct += 1
            self._total += 1

    def compute(self) -> dict[str, float]:
        accuracy = self._correct / self._total if self._total > 0 else 0.0
        return {"accuracy": accuracy}


@BaseMetric.register("binary_fbeta")
@BinaryClassificationMetric.register("fbeta")
class BinaryFBeta(BinaryClassificationMetric[BinaryLabelT], Generic[BinaryLabelT]):
    """Binary F-beta score with precision and recall.

    Computes F-beta score, precision, and recall for binary classification.
    F-beta is the weighted harmonic mean of precision and recall, where
    beta controls the weight of recall relative to precision.

    Args:
        beta: Weight of recall relative to precision. Common values:
            - `1.0`: F1 score (balanced)
            - `0.5`: F0.5 (emphasizes precision)
            - `2.0`: F2 (emphasizes recall)

    Returns:
        Dictionary with `"fbeta"`, `"precision"`, and `"recall"` metrics.

    Examples:
        >>> # F1 score (beta=1.0)
        >>> metric = BinaryFBeta(beta=1.0)
        >>> inputs = ClassificationInput(
        ...     predictions=[1, 1, 0, 1],
        ...     targets=[1, 0, 0, 1]
        ... )
        >>> metric.update(inputs)
        >>> result = metric.compute()
        >>> # {"fbeta": 0.67, "precision": 0.67, "recall": 1.0}

    """

    def __init__(self, beta: float = 1.0) -> None:
        self._beta = beta
        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

    def reset(self) -> None:
        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

    def update(self, inputs: BinaryClassificationInput[BinaryLabelT]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred, target in zip(predictions, targets):
            if pred == target == 1:
                self._true_positive += 1
            elif pred == 1 and target == 0:
                self._false_positive += 1
            elif pred == 0 and target == 1:
                self._false_negative += 1

    def compute(self) -> dict[str, float]:
        beta_sq = self._beta**2
        precision_denominator = self._true_positive + self._false_positive
        recall_denominator = self._true_positive + self._false_negative

        precision = self._true_positive / precision_denominator if precision_denominator > 0 else 0.0
        recall = self._true_positive / recall_denominator if recall_denominator > 0 else 0.0

        if precision + recall == 0:
            fbeta = 0.0
        else:
            fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

        return {"fbeta": fbeta, "precision": precision, "recall": recall}


@BaseMetric.register("binary_roc_auc")
@BinaryClassificationMetric.register("roc_auc")
class BinaryROCAUC(BinaryClassificationMetric[BinaryLabelT], Generic[BinaryLabelT]):
    """Binary ROC AUC (Area Under the Receiver Operating Characteristic Curve) metric.

    Computes the area under the ROC curve, which measures the model's ability
    to distinguish between positive and negative classes across all thresholds.
    ROC AUC ranges from 0 to 1, where 0.5 represents random guessing and 1.0
    represents perfect classification.

    Returns:
        Dictionary with `"roc_auc"` metric.

    Examples:
        >>> metric = BinaryROCAUC()
        >>> inputs = BinaryClassificationInputWithScores(
        ...     predictions=[1, 1, 0, 1],
        ...     scores=[0.9, 0.8, 0.3, 0.7],
        ...     targets=[1, 0, 0, 1]
        ... )
        >>> metric.update(inputs)
        >>> result = metric.compute()
        >>> # {"roc_auc": 0.75}

    """

    def __init__(self) -> None:
        self._scores: list[float] = []
        self._targets: list[int] = []

    def reset(self) -> None:
        self._scores = []
        self._targets = []

    def update(self, inputs: BinaryClassificationInput[BinaryLabelT]) -> None:
        assert inputs.scores is not None, "Scores are required for ROC AUC computation"

        scores = inputs.scores
        targets = inputs.targets
        assert len(scores) == len(targets), "Scores and targets must have the same length"

        for score, target in zip(scores, targets):
            self._scores.append(score)
            self._targets.append(1 if target == 1 or target is True else 0)

    def compute(self) -> dict[str, float]:
        if not self._scores:
            return {"roc_auc": 0.0}

        # Count total positives and negatives
        n_pos = sum(self._targets)
        n_neg = len(self._targets) - n_pos

        if n_pos == 0 or n_neg == 0:
            return {"roc_auc": 0.0}

        # Sort by scores in descending order, with ties broken by target (negatives first)
        sorted_pairs = sorted(zip(self._scores, self._targets), key=lambda x: (-x[0], x[1]))

        # Calculate ROC curve points and AUC
        tp = 0
        fp = 0
        prev_tp = 0
        prev_fp = 0
        prev_score = float("inf")
        auc = 0.0

        for score, target in sorted_pairs:
            # When score changes, add area for previous threshold
            if score != prev_score:
                # Add trapezoid area: width * average height
                auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
                prev_tp = tp
                prev_fp = fp
                prev_score = score

            if target == 1:
                tp += 1
            else:
                fp += 1

        # Add final trapezoid
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0

        # Normalize by total area
        auc /= n_pos * n_neg

        return {"roc_auc": auc}


@BaseMetric.register("binary_pr_auc")
@BinaryClassificationMetric.register("pr_auc")
class BinaryPRAUC(BinaryClassificationMetric[BinaryLabelT], Generic[BinaryLabelT]):
    """Binary PR AUC (Area Under the Precision-Recall Curve) metric.

    Computes the area under the Precision-Recall curve, which plots precision
    (y-axis) against recall (x-axis) at different classification thresholds.
    This metric is particularly useful for imbalanced datasets where ROC AUC
    might be overly optimistic.

    Unlike ROC AUC which uses false positive rate, PR AUC focuses on the
    positive class performance, making it more informative when the positive
    class is rare.

    Returns:
        Dictionary with `"pr_auc"` metric.

    Examples:
        >>> metric = BinaryPRAUC()
        >>> inputs = BinaryClassificationInputWithScores(
        ...     predictions=[1, 1, 0, 1],
        ...     scores=[0.9, 0.8, 0.3, 0.7],
        ...     targets=[1, 0, 0, 1]
        ... )
        >>> metric.update(inputs)
        >>> result = metric.compute()
        >>> # {"pr_auc": 0.833...}

    """

    def __init__(self) -> None:
        self._scores: list[float] = []
        self._targets: list[int] = []

    def reset(self) -> None:
        self._scores = []
        self._targets = []

    def update(self, inputs: BinaryClassificationInput[BinaryLabelT]) -> None:
        assert inputs.scores is not None, "Scores are required for PR AUC computation"

        scores = inputs.scores
        targets = inputs.targets
        assert len(scores) == len(targets), "Scores and targets must have the same length"

        for score, target in zip(scores, targets):
            self._scores.append(score)
            self._targets.append(1 if target == 1 or target is True else 0)

    def compute(self) -> dict[str, float]:
        if not self._scores:
            return {"pr_auc": 0.0}

        # Count total positives
        n_pos = sum(self._targets)
        if n_pos == 0:
            return {"pr_auc": 0.0}

        # Sort by scores in descending order
        sorted_pairs = sorted(zip(self._scores, self._targets), key=lambda x: (-x[0], x[1]))

        # Calculate precision and recall at each threshold
        precisions = []
        recalls = []

        tp = 0
        fp = 0

        for score, target in sorted_pairs:
            if target == 1:
                tp += 1
            else:
                fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_pos

            precisions.append(precision)
            recalls.append(recall)

        # Add point (0, 1) at the beginning if not already there
        if recalls[0] != 0:
            recalls.insert(0, 0.0)
            precisions.insert(0, precisions[0])

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(recalls) - 1):
            # Width in recall axis
            width = recalls[i + 1] - recalls[i]
            # Average height (precision)
            height = (precisions[i] + precisions[i + 1]) / 2.0
            auc += width * height

        return {"pr_auc": auc}


class MulticlassClassificationMetric(BaseMetric[ClassificationInput[LabelT]], Generic[LabelT]):
    """Base class for multiclass classification metrics.

    Multiclass metrics work with any number of classes and support
    both micro and macro averaging strategies.

    Type Parameters:
        LabelT: Type of labels (`int`, `str`, etc.).

    """

    Input: type[ClassificationInput[LabelT]] = ClassificationInput


@BaseMetric.register("multiclass_accuracy")
@MulticlassClassificationMetric.register("accuracy")
class MulticlassAccuracy(MulticlassClassificationMetric[LabelT], Generic[LabelT]):
    """Multiclass classification accuracy with averaging strategies.

    Computes accuracy for multiclass classification with support for
    micro (overall accuracy) and macro (per-class average) strategies.

    Args:
        average: Averaging strategy:
            - `"micro"`: Overall accuracy across all samples
            - `"macro"`: Average of per-class accuracies

    Examples:
        >>> # Micro averaging (overall accuracy)
        >>> metric = MulticlassAccuracy(average="micro")
        >>> inputs = ClassificationInput(
        ...     predictions=[0, 1, 2, 1],
        ...     targets=[0, 1, 1, 1]
        ... )
        >>> metric.update(inputs)
        >>> metric.compute()  # {"accuracy": 0.75}
        >>>
        >>> # Macro averaging (per-class average)
        >>> metric = MulticlassAccuracy(average="macro")
        >>> metric.update(inputs)
        >>> metric.compute()  # Average of class-wise accuracies

    """

    def __init__(self, average: Literal["micro", "macro"] = "micro") -> None:
        self._average = average
        self._correct: dict[LabelT, int] = defaultdict(int)
        self._total: dict[LabelT, int] = defaultdict(int)

    def reset(self) -> None:
        self._correct = defaultdict(int)
        self._total = defaultdict(int)

    def update(self, inputs: ClassificationInput[LabelT]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred, target in zip(predictions, targets):
            if pred == target:
                self._correct[target] += 1
            self._total[target] += 1

    def compute(self) -> dict[str, float]:
        if self._average == "micro":
            total_correct = sum(self._correct.values())
            total_count = sum(self._total.values())
            accuracy = total_correct / total_count if total_count > 0 else 0.0
            return {"accuracy": accuracy}
        elif self._average == "macro":
            accuracies = []
            for label in self._total.keys():
                correct = self._correct[label]
                total = self._total[label]
                accuracies.append(correct / total if total > 0 else 0.0)
            macro_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            return {"accuracy": macro_accuracy}
        else:
            raise ValueError(f"Unknown average type: {self._average}")


@BaseMetric.register("multiclass_fbeta")
@MulticlassClassificationMetric.register("fbeta")
class MulticlassFBeta(MulticlassClassificationMetric[LabelT], Generic[LabelT]):
    """Multiclass F-beta score with precision and recall.

    Computes F-beta, precision, and recall for multiclass classification
    with support for micro and macro averaging.

    Args:
        beta: Weight of recall relative to precision (default: `1.0` for F1).
        average: Averaging strategy:
            - `"micro"`: Compute globally across all classes
            - `"macro"`: Compute per-class then average

    Returns:
        Dictionary with "fbeta", "precision", and "recall" metrics.

    Examples:
        >>> metric = MulticlassFBeta(beta=1.0, average="macro")
        >>> inputs = ClassificationInput(
        ...     predictions=[0, 1, 2, 1],
        ...     targets=[0, 1, 1, 1]
        ... )
        >>> metric.update(inputs)
        >>> metric.compute()
        >>> # {"fbeta": ..., "precision": ..., "recall": ...}

    """

    def __init__(self, beta: float = 1.0, average: Literal["micro", "macro"] = "micro") -> None:
        self._beta = beta
        self._average = average
        self._true_positive: dict[LabelT, int] = defaultdict(int)
        self._false_positive: dict[LabelT, int] = defaultdict(int)
        self._false_negative: dict[LabelT, int] = defaultdict(int)

    def reset(self) -> None:
        self._true_positive = defaultdict(int)
        self._false_positive = defaultdict(int)
        self._false_negative = defaultdict(int)

    def update(self, inputs: ClassificationInput[LabelT]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred, target in zip(predictions, targets):
            if pred == target:
                self._true_positive[target] += 1
            else:
                self._false_positive[pred] += 1
                self._false_negative[target] += 1

    def compute(self) -> dict[str, float]:
        beta_sq = self._beta**2

        if self._average == "micro":
            total_true_positive = sum(self._true_positive.values())
            total_false_positive = sum(self._false_positive.values())
            total_false_negative = sum(self._false_negative.values())

            precision_denominator = total_true_positive + total_false_positive
            recall_denominator = total_true_positive + total_false_negative

            precision = total_true_positive / precision_denominator if precision_denominator > 0 else 0.0
            recall = total_true_positive / recall_denominator if recall_denominator > 0 else 0.0

            if precision + recall == 0:
                fbeta = 0.0
            else:
                fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

            return {"fbeta": fbeta, "precision": precision, "recall": recall}

        elif self._average == "macro":
            fbetas = []
            precisions = []
            recalls = []

            for label in (
                set(self._true_positive.keys()).union(self._false_positive.keys()).union(self._false_negative.keys())
            ):
                tp = self._true_positive[label]
                fp = self._false_positive[label]
                fn = self._false_negative[label]

                precision_denominator = tp + fp
                recall_denominator = tp + fn

                precision = tp / precision_denominator if precision_denominator > 0 else 0.0
                recall = tp / recall_denominator if recall_denominator > 0 else 0.0

                if precision + recall == 0:
                    fbeta = 0.0
                else:
                    fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

                fbetas.append(fbeta)
                precisions.append(precision)
                recalls.append(recall)
            macro_fbeta = sum(fbetas) / len(fbetas) if fbetas else 0.0
            macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
            macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
            return {"fbeta": macro_fbeta, "precision": macro_precision, "recall": macro_recall}
        else:
            raise ValueError(f"Unknown average type: {self._average}")


class MultilabelClassificationMetric(BaseMetric[ClassificationInput[Sequence[LabelT]]], Generic[LabelT]):
    Input: type[ClassificationInput[Sequence[LabelT]]] = ClassificationInput


@BaseMetric.register("multilabel_accuracy")
@MultilabelClassificationMetric.register("accuracy")
class MultilabelAccuracy(MultilabelClassificationMetric[LabelT], Generic[LabelT]):
    def __init__(self, average: Literal["micro", "macro"] = "micro") -> None:
        self._average = average
        self._correct: dict[LabelT, int] = defaultdict(int)
        self._total: dict[LabelT, int] = defaultdict(int)

    def reset(self) -> None:
        self._correct = defaultdict(int)
        self._total = defaultdict(int)

    def update(self, inputs: ClassificationInput[Sequence[LabelT]]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred_labels, target_labels in zip(predictions, targets):
            pred_set = set(pred_labels)
            target_set = set(target_labels)
            for label in target_set.union(pred_set):
                if label in target_set and label in pred_set:
                    self._correct[label] += 1
                self._total[label] += 1

    def compute(self) -> dict[str, float]:
        if self._average == "micro":
            total_correct = sum(self._correct.values())
            total_count = sum(self._total.values())
            accuracy = total_correct / total_count if total_count > 0 else 0.0
            return {"accuracy": accuracy}
        elif self._average == "macro":
            accuracies = []
            for label in self._total.keys():
                correct = self._correct[label]
                total = self._total[label]
                accuracies.append(correct / total if total > 0 else 0.0)
            macro_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
            return {"accuracy": macro_accuracy}
        else:
            raise ValueError(f"Unknown average type: {self._average}")


@BaseMetric.register("multilabel_fbeta")
@MultilabelClassificationMetric.register("fbeta")
class MultilabelFBeta(MultilabelClassificationMetric[LabelT], Generic[LabelT]):
    def __init__(self, beta: float = 1.0, average: Literal["micro", "macro"] = "micro") -> None:
        self._beta = beta
        self._average = average
        self._true_positive: dict[LabelT, int] = defaultdict(int)
        self._false_positive: dict[LabelT, int] = defaultdict(int)
        self._false_negative: dict[LabelT, int] = defaultdict(int)

    def reset(self) -> None:
        self._true_positive = defaultdict(int)
        self._false_positive = defaultdict(int)
        self._false_negative = defaultdict(int)

    def update(self, inputs: ClassificationInput[Sequence[LabelT]]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred_labels, target_labels in zip(predictions, targets):
            pred_set = set(pred_labels)
            target_set = set(target_labels)
            for label in target_set.union(pred_set):
                if label in target_set and label in pred_set:
                    self._true_positive[label] += 1
                elif label in pred_set and label not in target_set:
                    self._false_positive[label] += 1
                elif label in target_set and label not in pred_set:
                    self._false_negative[label] += 1

    def compute(self) -> dict[str, float]:
        beta_sq = self._beta**2

        if self._average == "micro":
            total_true_positive = sum(self._true_positive.values())
            total_false_positive = sum(self._false_positive.values())
            total_false_negative = sum(self._false_negative.values())

            precision_denominator = total_true_positive + total_false_positive
            recall_denominator = total_true_positive + total_false_negative

            precision = total_true_positive / precision_denominator if precision_denominator > 0 else 0.0
            recall = total_true_positive / recall_denominator if recall_denominator > 0 else 0.0

            if precision + recall == 0:
                fbeta = 0.0
            else:
                fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

            return {"fbeta": fbeta, "precision": precision, "recall": recall}
        elif self._average == "macro":
            fbetas = []
            precisions = []
            recalls = []

            for label in (
                set(self._true_positive.keys()).union(self._false_positive.keys()).union(self._false_negative.keys())
            ):
                tp = self._true_positive[label]
                fp = self._false_positive[label]
                fn = self._false_negative[label]

                precision_denominator = tp + fp
                recall_denominator = tp + fn

                precision = tp / precision_denominator if precision_denominator > 0 else 0.0
                recall = tp / recall_denominator if recall_denominator > 0 else 0.0

                if precision + recall == 0:
                    fbeta = 0.0
                else:
                    fbeta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

                fbetas.append(fbeta)
                precisions.append(precision)
                recalls.append(recall)
            macro_fbeta = sum(fbetas) / len(fbetas) if fbetas else 0.0
            macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
            macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
            return {"fbeta": macro_fbeta, "precision": macro_precision, "recall": macro_recall}
        else:
            raise ValueError(f"Unknown average type: {self._average}")


@dataclasses.dataclass
class RegressionInput:
    predictions: Sequence[float]
    targets: Sequence[float]


class RegressionMetric(BaseMetric[RegressionInput]):
    Input: type[RegressionInput] = RegressionInput


@BaseMetric.register("mean_squared_error")
@RegressionMetric.register("mean_squared_error")
class MeanSquaredError(RegressionMetric):
    def __init__(self) -> None:
        self._squared_error = 0.0
        self._count = 0

    def reset(self) -> None:
        self._squared_error = 0.0
        self._count = 0

    def update(self, inputs: RegressionInput) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred, target in zip(predictions, targets):
            self._squared_error += (pred - target) ** 2
            self._count += 1

    def compute(self) -> dict[str, float]:
        mse = self._squared_error / self._count if self._count > 0 else 0.0
        return {"mean_squared_error": mse}


@BaseMetric.register("mean_absolute_error")
@RegressionMetric.register("mean_absolute_error")
class MeanAbsoluteError(RegressionMetric):
    def __init__(self) -> None:
        self._absolute_error = 0.0
        self._count = 0

    def reset(self) -> None:
        self._absolute_error = 0.0
        self._count = 0

    def update(self, inputs: RegressionInput) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred, target in zip(predictions, targets):
            self._absolute_error += abs(pred - target)
            self._count += 1

    def compute(self) -> dict[str, float]:
        mae = self._absolute_error / self._count if self._count > 0 else 0.0
        return {"mean_absolute_error": mae}


@dataclasses.dataclass
class RankingInput(Generic[LabelT]):
    predictions: Sequence[Mapping[LabelT, float]]
    targets: Sequence[Sequence[LabelT]]


class RankingMetric(BaseMetric[RankingInput[LabelT]], Generic[LabelT]):
    Input: type[RankingInput[LabelT]] = RankingInput


@BaseMetric.register("mean_average_precision")
@RankingMetric.register("mean_average_precision")
class MeanAveragePrecision(RankingMetric[LabelT], Generic[LabelT]):
    def __init__(self) -> None:
        self._average_precisions: list[float] = []

    def reset(self) -> None:
        self._average_precisions = []

    def update(self, inputs: RankingInput[LabelT]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred_scores, target_labels in zip(predictions, targets):
            sorted_labels = sorted(pred_scores.keys(), key=lambda x: pred_scores[x], reverse=True)
            relevant_set = set(target_labels)

            num_relevant = 0
            precision_sum = 0.0

            for rank, label in enumerate(sorted_labels, start=1):
                if label in relevant_set:
                    num_relevant += 1
                    precision_sum += num_relevant / rank

            average_precision = precision_sum / len(relevant_set) if relevant_set else 0.0
            self._average_precisions.append(average_precision)

    def compute(self) -> dict[str, float]:
        mean_ap = sum(self._average_precisions) / len(self._average_precisions) if self._average_precisions else 0.0
        return {"mean_average_precision": mean_ap}


@BaseMetric.register("ndcg")
@RankingMetric.register("ndcg")
class NDCG(RankingMetric[LabelT], Generic[LabelT]):
    def __init__(self, k: int = 10) -> None:
        self._k = k
        self._ndcgs: list[float] = []

    def reset(self) -> None:
        self._ndcgs = []

    def update(self, inputs: RankingInput[LabelT]) -> None:
        predictions = inputs.predictions
        targets = inputs.targets
        assert len(predictions) == len(targets), "Predictions and targets must have the same length"

        for pred_scores, target_labels in zip(predictions, targets):
            sorted_labels = sorted(pred_scores.keys(), key=lambda x: pred_scores[x], reverse=True)
            relevant_set = set(target_labels)

            dcg = 0.0
            for rank, label in enumerate(sorted_labels[: self._k], start=1):
                if label in relevant_set:
                    dcg += 1 / math.log2(rank + 1)

            ideal_dcg = 0.0
            for rank in range(1, min(len(relevant_set), self._k) + 1):
                ideal_dcg += 1 / math.log2(rank + 1)

            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            self._ndcgs.append(ndcg)

    def compute(self) -> dict[str, float]:
        mean_ndcg = sum(self._ndcgs) / len(self._ndcgs) if self._ndcgs else 0.0
        return {"ndcg": mean_ndcg}
