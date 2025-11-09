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
    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, inputs: _T) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self) -> dict[str, float]:
        raise NotImplementedError()

    def __call__(self, inputs: _T) -> dict[str, float]:
        self.update(inputs)
        return self.compute()


@BaseMetric.register("empty")
class EmptyMetric(BaseMetric[Any]):
    def reset(self) -> None:
        pass

    def update(self, inputs: Any) -> None:
        pass

    def compute(self) -> dict[str, float]:
        return {}


@BaseMetric.register("average")
class Average(BaseMetric[Sequence[float]]):
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
    predictions: Sequence[_T]
    targets: Sequence[_T]


@BaseMetric.register("binary_accuracy")
class BinaryAccuracy(BaseMetric[ClassificationInput[BinaryLabelT]], Generic[BinaryLabelT]):
    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, inputs: ClassificationInput[BinaryLabelT]) -> None:
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


@BaseMetric.register("multiclass_accuracy")
class MulticlassAccuracy(BaseMetric[ClassificationInput[LabelT]], Generic[LabelT]):
    Input: type[ClassificationInput[LabelT]] = ClassificationInput

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


@BaseMetric.register("multilabel_accuracy")
class MultilabelAccuracy(BaseMetric[ClassificationInput[Sequence[LabelT]]], Generic[LabelT]):
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


@BaseMetric.register("binary_fbeta")
class BinaryFBeta(BaseMetric[ClassificationInput[BinaryLabelT]], Generic[BinaryLabelT]):
    def __init__(self, beta: float = 1.0) -> None:
        self._beta = beta
        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

    def reset(self) -> None:
        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0

    def update(self, inputs: ClassificationInput[BinaryLabelT]) -> None:
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


@BaseMetric.register("multiclass_fbeta")
class MulticlassFBeta(BaseMetric[ClassificationInput[LabelT]], Generic[LabelT]):
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


@BaseMetric.register("multilabel_fbeta")
class MultilabelFBeta(BaseMetric[ClassificationInput[Sequence[LabelT]]], Generic[LabelT]):
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


@BaseMetric.register("mean_squared_error")
class MeanSquaredError(BaseMetric[RegressionInput]):
    Input = RegressionInput

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
class MeanAbsoluteError(BaseMetric[RegressionInput]):
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


@BaseMetric.register("mean_average_precision")
class MeanAveragePrecision(BaseMetric[RankingInput[LabelT]], Generic[LabelT]):
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
class NDCG(BaseMetric[RankingInput[LabelT]], Generic[LabelT]):
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
