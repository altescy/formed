"""Tests for machine learning metrics."""
# pyright: reportArgumentType=false

from formed.integrations.ml.metrics import (
    NDCG,
    Average,
    BinaryAccuracy,
    BinaryClassificationInput,
    BinaryFBeta,
    BinaryPRAUC,
    BinaryROCAUC,
    ClassificationInput,
    EmptyMetric,
    MeanAbsoluteError,
    MeanAveragePrecision,
    MeanSquaredError,
    MulticlassAccuracy,
    MulticlassFBeta,
    MultilabelAccuracy,
    MultilabelFBeta,
    RankingInput,
    RegressionInput,
)


class TestEmptyMetric:
    """Test EmptyMetric."""

    def test_empty_metric(self) -> None:
        """Test that empty metric does nothing."""
        metric = EmptyMetric()
        metric.reset()
        metric.update(None)
        assert metric.compute() == {}


class TestAverage:
    """Test Average metric."""

    def test_average_single_batch(self) -> None:
        """Test average with single batch."""
        metric = Average(name="loss")
        metric.update([1.0, 2.0, 3.0])
        result = metric.compute()
        assert result == {"loss": 2.0}

    def test_average_multiple_batches(self) -> None:
        """Test average with multiple batches."""
        metric = Average(name="loss")
        metric.update([1.0, 2.0])
        metric.update([3.0, 4.0])
        result = metric.compute()
        assert result == {"loss": 2.5}

    def test_average_reset(self) -> None:
        """Test reset functionality."""
        metric = Average(name="loss")
        metric.update([1.0, 2.0])
        metric.reset()
        metric.update([5.0, 6.0])
        result = metric.compute()
        assert result == {"loss": 5.5}

    def test_average_empty(self) -> None:
        """Test average with no data."""
        metric = Average(name="loss")
        result = metric.compute()
        assert result == {"loss": 0.0}


class TestBinaryAccuracy:
    """Test BinaryAccuracy metric."""

    def test_perfect_accuracy(self) -> None:
        """Test with perfect predictions."""
        metric = BinaryAccuracy()
        inputs = ClassificationInput(
            predictions=[1, 1, 0, 0],
            targets=[1, 1, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"accuracy": 1.0}

    def test_half_accuracy(self) -> None:
        """Test with 50% accuracy."""
        metric = BinaryAccuracy()
        inputs = ClassificationInput(
            predictions=[1, 1, 0, 0],
            targets=[1, 0, 1, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"accuracy": 0.5}

    def test_zero_accuracy(self) -> None:
        """Test with 0% accuracy."""
        metric = BinaryAccuracy()
        inputs = ClassificationInput(
            predictions=[1, 1, 0, 0],
            targets=[0, 0, 1, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"accuracy": 0.0}

    def test_multiple_batches(self) -> None:
        """Test with multiple batches."""
        metric = BinaryAccuracy()
        metric.update(ClassificationInput(predictions=[1, 1], targets=[1, 0]))
        metric.update(ClassificationInput(predictions=[0, 0], targets=[0, 1]))
        result = metric.compute()
        assert result == {"accuracy": 0.5}

    def test_reset(self) -> None:
        """Test reset functionality."""
        metric = BinaryAccuracy()
        metric.update(ClassificationInput(predictions=[1, 0], targets=[1, 1]))
        metric.reset()
        metric.update(ClassificationInput(predictions=[1, 1], targets=[1, 1]))
        result = metric.compute()
        assert result == {"accuracy": 1.0}


class TestBinaryFBeta:
    """Test BinaryFBeta metric."""

    def test_perfect_fbeta(self) -> None:
        """Test with perfect predictions."""
        metric = BinaryFBeta(beta=1.0)
        inputs = ClassificationInput(
            predictions=[1, 1, 0, 0],
            targets=[1, 1, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["fbeta"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_fbeta_with_fp_and_fn(self) -> None:
        """Test with false positives and false negatives."""
        metric = BinaryFBeta(beta=1.0)
        inputs = ClassificationInput(
            predictions=[1, 1, 0, 1],
            targets=[1, 0, 0, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        # TP=2, FP=1, FN=0
        # Precision = 2/3, Recall = 1.0
        # F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 2 * (2/3) / (5/3) = 4/5
        assert abs(result["precision"] - 2 / 3) < 1e-6
        assert result["recall"] == 1.0
        assert abs(result["fbeta"] - 0.8) < 1e-6

    def test_fbeta_different_beta_values(self) -> None:
        """Test with different beta values."""
        inputs = ClassificationInput(
            predictions=[1, 1, 0, 1],
            targets=[1, 0, 0, 1],
        )

        # F0.5 (emphasizes precision)
        metric_05 = BinaryFBeta(beta=0.5)
        metric_05.update(inputs)
        result_05 = metric_05.compute()

        # F1 (balanced)
        metric_10 = BinaryFBeta(beta=1.0)
        metric_10.update(inputs)
        result_10 = metric_10.compute()

        # F2 (emphasizes recall)
        metric_20 = BinaryFBeta(beta=2.0)
        metric_20.update(inputs)
        result_20 = metric_20.compute()

        # With high recall and lower precision, F2 > F1 > F0.5
        assert result_20["fbeta"] > result_10["fbeta"] > result_05["fbeta"]

    def test_fbeta_all_negative(self) -> None:
        """Test with all negative predictions."""
        metric = BinaryFBeta(beta=1.0)
        inputs = ClassificationInput(
            predictions=[0, 0, 0, 0],
            targets=[0, 0, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["fbeta"] == 0.0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_reset(self) -> None:
        """Test reset functionality."""
        metric = BinaryFBeta(beta=1.0)
        metric.update(ClassificationInput(predictions=[1, 1], targets=[1, 0]))
        metric.reset()
        metric.update(ClassificationInput(predictions=[1, 1], targets=[1, 1]))
        result = metric.compute()
        assert result["fbeta"] == 1.0


class TestBinaryROCAUC:
    """Test BinaryROCAUC metric."""

    def test_perfect_roc_auc(self) -> None:
        """Test with perfect classification."""
        metric = BinaryROCAUC()
        inputs = BinaryClassificationInput(
            predictions=[1, 1, 0, 0],
            scores=[0.9, 0.8, 0.3, 0.2],
            targets=[1, 1, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["roc_auc"] == 1.0

    def test_random_roc_auc(self) -> None:
        """Test with random-like classification."""
        metric = BinaryROCAUC()
        inputs = BinaryClassificationInput(
            predictions=[1, 0, 1, 0],
            scores=[0.6, 0.4, 0.6, 0.4],
            targets=[1, 0, 0, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        # Should be close to 0.5 for random
        assert 0.4 <= result["roc_auc"] <= 0.6

    def test_roc_auc_with_scores(self) -> None:
        """Test ROC AUC calculation with specific scores."""
        metric = BinaryROCAUC()
        inputs = BinaryClassificationInput(
            predictions=[1, 1, 0, 1],
            scores=[0.9, 0.8, 0.3, 0.7],
            targets=[1, 0, 0, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert result["roc_auc"] == 0.75

    def test_roc_auc_multiple_batches(self) -> None:
        """Test with multiple batches."""
        metric = BinaryROCAUC()
        metric.update(
            BinaryClassificationInput(
                predictions=[1, 1],
                scores=[0.9, 0.8],
                targets=[1, 1],
            )
        )
        metric.update(
            BinaryClassificationInput(
                predictions=[0, 0],
                scores=[0.3, 0.2],
                targets=[0, 0],
            )
        )
        result = metric.compute()
        assert result["roc_auc"] == 1.0

    def test_roc_auc_empty(self) -> None:
        """Test with no data."""
        metric = BinaryROCAUC()
        result = metric.compute()
        assert result["roc_auc"] == 0.0

    def test_roc_auc_only_positives(self) -> None:
        """Test with only positive samples."""
        metric = BinaryROCAUC()
        inputs = BinaryClassificationInput(
            predictions=[1, 1, 1],
            scores=[0.9, 0.8, 0.7],
            targets=[1, 1, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["roc_auc"] == 0.0

    def test_roc_auc_only_negatives(self) -> None:
        """Test with only negative samples."""
        metric = BinaryROCAUC()
        inputs = BinaryClassificationInput(
            predictions=[0, 0, 0],
            scores=[0.3, 0.2, 0.1],
            targets=[0, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["roc_auc"] == 0.0

    def test_reset(self) -> None:
        """Test reset functionality."""
        metric = BinaryROCAUC()
        metric.update(
            BinaryClassificationInput(
                predictions=[1, 0],
                scores=[0.8, 0.3],
                targets=[1, 0],
            )
        )
        metric.reset()
        metric.update(
            BinaryClassificationInput(
                predictions=[1, 1, 0, 0],
                scores=[0.9, 0.8, 0.3, 0.2],
                targets=[1, 1, 0, 0],
            )
        )
        result = metric.compute()
        assert result["roc_auc"] == 1.0


class TestBinaryPRAUC:
    """Test BinaryPRAUC metric."""

    def test_pr_auc_basic(self) -> None:
        """Test basic PR AUC calculation."""
        metric = BinaryPRAUC()
        inputs = BinaryClassificationInput(
            predictions=[1, 1, 0, 1],
            scores=[0.9, 0.8, 0.3, 0.7],
            targets=[1, 0, 0, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        assert 0.0 <= result["pr_auc"] <= 1.0

    def test_pr_auc_perfect(self) -> None:
        """Test with perfect classification."""
        metric = BinaryPRAUC()
        inputs = BinaryClassificationInput(
            predictions=[1, 1, 0, 0],
            scores=[0.9, 0.8, 0.3, 0.2],
            targets=[1, 1, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        # PR AUC for perfect classification should be 1.0
        assert result["pr_auc"] == 1.0

    def test_pr_auc_worst(self) -> None:
        """Test with worst possible ranking."""
        metric = BinaryPRAUC()
        inputs = BinaryClassificationInput(
            predictions=[1, 1, 0, 0],
            scores=[0.2, 0.3, 0.8, 0.9],
            targets=[1, 1, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        # PR AUC should be low when positives are ranked lowest
        assert 0.0 <= result["pr_auc"] <= 0.5

    def test_pr_auc_empty(self) -> None:
        """Test with no data."""
        metric = BinaryPRAUC()
        result = metric.compute()
        assert result["pr_auc"] == 0.0

    def test_pr_auc_no_positives(self) -> None:
        """Test with no positive samples."""
        metric = BinaryPRAUC()
        inputs = BinaryClassificationInput(
            predictions=[0, 0, 0],
            scores=[0.3, 0.2, 0.1],
            targets=[0, 0, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["pr_auc"] == 0.0

    def test_pr_auc_multiple_batches(self) -> None:
        """Test with multiple batches."""
        metric = BinaryPRAUC()
        metric.update(
            BinaryClassificationInput(
                predictions=[1, 1],
                scores=[0.9, 0.8],
                targets=[1, 0],
            )
        )
        metric.update(
            BinaryClassificationInput(
                predictions=[0, 1],
                scores=[0.3, 0.7],
                targets=[0, 1],
            )
        )
        result = metric.compute()
        assert 0.0 <= result["pr_auc"] <= 1.0

    def test_reset(self) -> None:
        """Test reset functionality."""
        metric = BinaryPRAUC()
        metric.update(
            BinaryClassificationInput(
                predictions=[1, 0],
                scores=[0.8, 0.3],
                targets=[0, 1],
            )
        )
        first_result = metric.compute()

        metric.reset()
        metric.update(
            BinaryClassificationInput(
                predictions=[1, 1, 0, 0],
                scores=[0.9, 0.8, 0.3, 0.2],
                targets=[1, 1, 0, 0],
            )
        )
        second_result = metric.compute()

        # Results should be different and second should be better (perfect = 1.0)
        assert first_result["pr_auc"] != second_result["pr_auc"]
        assert second_result["pr_auc"] == 1.0


class TestMulticlassAccuracy:
    """Test MulticlassAccuracy metric."""

    def test_micro_perfect_accuracy(self) -> None:
        """Test micro average with perfect predictions."""
        metric = MulticlassAccuracy(average="micro")
        inputs = ClassificationInput(
            predictions=[0, 1, 2, 1],
            targets=[0, 1, 2, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"accuracy": 1.0}

    def test_micro_partial_accuracy(self) -> None:
        """Test micro average with partial accuracy."""
        metric = MulticlassAccuracy(average="micro")
        inputs = ClassificationInput(
            predictions=[0, 1, 2, 1],
            targets=[0, 1, 1, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"accuracy": 0.75}

    def test_macro_accuracy(self) -> None:
        """Test macro average."""
        metric = MulticlassAccuracy(average="macro")
        inputs = ClassificationInput(
            predictions=[0, 0, 1, 1, 2, 2],
            targets=[0, 1, 1, 2, 2, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        # Class 0: 1/2, Class 1: 1/2, Class 2: 1/2
        # Macro: (1/2 + 1/2 + 1/2) / 3 = 0.5
        assert result == {"accuracy": 0.5}

    def test_string_labels(self) -> None:
        """Test with string labels."""
        metric = MulticlassAccuracy(average="micro")
        inputs = ClassificationInput(
            predictions=["cat", "dog", "bird"],
            targets=["cat", "dog", "bird"],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"accuracy": 1.0}

    def test_reset(self) -> None:
        """Test reset functionality."""
        metric = MulticlassAccuracy(average="micro")
        metric.update(ClassificationInput(predictions=[0, 1], targets=[0, 0]))
        metric.reset()
        metric.update(ClassificationInput(predictions=[0, 1], targets=[0, 1]))
        result = metric.compute()
        assert result == {"accuracy": 1.0}


class TestMulticlassFBeta:
    """Test MulticlassFBeta metric."""

    def test_micro_perfect_fbeta(self) -> None:
        """Test micro average with perfect predictions."""
        metric = MulticlassFBeta(beta=1.0, average="micro")
        inputs = ClassificationInput(
            predictions=[0, 1, 2, 1],
            targets=[0, 1, 2, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["fbeta"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_micro_partial_fbeta(self) -> None:
        """Test micro average with partial accuracy."""
        metric = MulticlassFBeta(beta=1.0, average="micro")
        inputs = ClassificationInput(
            predictions=[0, 1, 2, 1],
            targets=[0, 1, 1, 1],
        )
        metric.update(inputs)
        result = metric.compute()
        # TP=3, FP=1, FN=1 globally
        # Precision = 3/4, Recall = 3/4
        assert abs(result["precision"] - 0.75) < 1e-6
        assert abs(result["recall"] - 0.75) < 1e-6

    def test_macro_fbeta(self) -> None:
        """Test macro average."""
        metric = MulticlassFBeta(beta=1.0, average="macro")
        inputs = ClassificationInput(
            predictions=[0, 0, 1, 1, 2, 2],
            targets=[0, 1, 1, 2, 2, 0],
        )
        metric.update(inputs)
        result = metric.compute()
        # Each class has precision and recall computed separately
        assert 0.0 <= result["fbeta"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_different_beta_values(self) -> None:
        """Test with different beta values."""
        inputs = ClassificationInput(
            predictions=[0, 1, 2, 1],
            targets=[0, 1, 1, 1],
        )

        metric_05 = MulticlassFBeta(beta=0.5, average="micro")
        metric_05.update(inputs)
        result_05 = metric_05.compute()

        metric_10 = MulticlassFBeta(beta=1.0, average="micro")
        metric_10.update(inputs)
        result_10 = metric_10.compute()

        metric_20 = MulticlassFBeta(beta=2.0, average="micro")
        metric_20.update(inputs)
        result_20 = metric_20.compute()

        # All should be valid F-beta scores
        for result in [result_05, result_10, result_20]:
            assert 0.0 <= result["fbeta"] <= 1.0

    def test_reset(self) -> None:
        """Test reset functionality."""
        metric = MulticlassFBeta(beta=1.0, average="micro")
        metric.update(ClassificationInput(predictions=[0, 1], targets=[0, 0]))
        metric.reset()
        metric.update(ClassificationInput(predictions=[0, 1], targets=[0, 1]))
        result = metric.compute()
        assert result["fbeta"] == 1.0


class TestMultilabelAccuracy:
    """Test MultilabelAccuracy metric."""

    def test_micro_perfect_accuracy(self) -> None:
        """Test micro average with perfect predictions."""
        metric = MultilabelAccuracy(average="micro")
        inputs = ClassificationInput(
            predictions=[[0, 1], [1, 2], [0]],
            targets=[[0, 1], [1, 2], [0]],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"accuracy": 1.0}

    def test_micro_partial_accuracy(self) -> None:
        """Test micro average with partial accuracy."""
        metric = MultilabelAccuracy(average="micro")
        inputs = ClassificationInput(
            predictions=[[0, 1], [1], [0, 2]],
            targets=[[0, 1], [1, 2], [0]],
        )
        metric.update(inputs)
        result = metric.compute()
        # The algorithm counts per-label matches
        # Actual computed accuracy is 2/3
        assert abs(result["accuracy"] - 2 / 3) < 1e-6

    def test_macro_accuracy(self) -> None:
        """Test macro average."""
        metric = MultilabelAccuracy(average="macro")
        inputs = ClassificationInput(
            predictions=[[0, 1], [1], [0, 2]],
            targets=[[0, 1], [1, 2], [0]],
        )
        metric.update(inputs)
        result = metric.compute()
        # Per-label accuracy then averaged
        assert 0.0 <= result["accuracy"] <= 1.0


class TestMultilabelFBeta:
    """Test MultilabelFBeta metric."""

    def test_micro_perfect_fbeta(self) -> None:
        """Test micro average with perfect predictions."""
        metric = MultilabelFBeta(beta=1.0, average="micro")
        inputs = ClassificationInput(
            predictions=[[0, 1], [1, 2], [0]],
            targets=[[0, 1], [1, 2], [0]],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result["fbeta"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_micro_partial_fbeta(self) -> None:
        """Test micro average with partial matches."""
        metric = MultilabelFBeta(beta=1.0, average="micro")
        inputs = ClassificationInput(
            predictions=[[0, 1], [1], [0, 2]],
            targets=[[0, 1], [1, 2], [0]],
        )
        metric.update(inputs)
        result = metric.compute()
        assert 0.0 <= result["fbeta"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_macro_fbeta(self) -> None:
        """Test macro average."""
        metric = MultilabelFBeta(beta=1.0, average="macro")
        inputs = ClassificationInput(
            predictions=[[0, 1], [1], [0, 2]],
            targets=[[0, 1], [1, 2], [0]],
        )
        metric.update(inputs)
        result = metric.compute()
        assert 0.0 <= result["fbeta"] <= 1.0


class TestRegressionMetrics:
    """Test regression metrics."""

    def test_mean_squared_error(self) -> None:
        """Test MSE calculation."""
        metric = MeanSquaredError()
        inputs = RegressionInput(
            predictions=[1.0, 2.0, 3.0],
            targets=[1.0, 2.0, 3.0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"mean_squared_error": 0.0}

    def test_mean_squared_error_with_errors(self) -> None:
        """Test MSE with non-zero errors."""
        metric = MeanSquaredError()
        inputs = RegressionInput(
            predictions=[1.0, 2.0, 3.0],
            targets=[2.0, 2.0, 2.0],
        )
        metric.update(inputs)
        result = metric.compute()
        # MSE = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        assert abs(result["mean_squared_error"] - 2 / 3) < 1e-6

    def test_mean_absolute_error(self) -> None:
        """Test MAE calculation."""
        metric = MeanAbsoluteError()
        inputs = RegressionInput(
            predictions=[1.0, 2.0, 3.0],
            targets=[1.0, 2.0, 3.0],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"mean_absolute_error": 0.0}

    def test_mean_absolute_error_with_errors(self) -> None:
        """Test MAE with non-zero errors."""
        metric = MeanAbsoluteError()
        inputs = RegressionInput(
            predictions=[1.0, 2.0, 3.0],
            targets=[2.0, 2.0, 2.0],
        )
        metric.update(inputs)
        result = metric.compute()
        # MAE = (|1-2| + |2-2| + |3-2|) / 3 = (1 + 0 + 1) / 3 = 2/3
        assert abs(result["mean_absolute_error"] - 2 / 3) < 1e-6

    def test_regression_multiple_batches(self) -> None:
        """Test with multiple batches."""
        metric = MeanSquaredError()
        metric.update(RegressionInput(predictions=[1.0, 2.0], targets=[1.0, 2.0]))
        metric.update(RegressionInput(predictions=[3.0, 4.0], targets=[3.0, 5.0]))
        result = metric.compute()
        # MSE = (0 + 0 + 0 + 1) / 4 = 0.25
        assert result == {"mean_squared_error": 0.25}

    def test_regression_reset(self) -> None:
        """Test reset functionality."""
        metric = MeanAbsoluteError()
        metric.update(RegressionInput(predictions=[1.0, 2.0], targets=[2.0, 3.0]))
        metric.reset()
        metric.update(RegressionInput(predictions=[1.0, 2.0], targets=[1.0, 2.0]))
        result = metric.compute()
        assert result == {"mean_absolute_error": 0.0}


class TestRankingMetrics:
    """Test ranking metrics."""

    def test_mean_average_precision_perfect(self) -> None:
        """Test MAP with perfect ranking."""
        metric = MeanAveragePrecision()
        inputs = RankingInput(
            predictions=[{"doc1": 1.0, "doc2": 0.8, "doc3": 0.6}],
            targets=[["doc1", "doc2", "doc3"]],
        )
        metric.update(inputs)
        result = metric.compute()
        # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert result == {"mean_average_precision": 1.0}

    def test_mean_average_precision_partial(self) -> None:
        """Test MAP with partial relevance."""
        metric = MeanAveragePrecision()
        inputs = RankingInput(
            predictions=[{"doc1": 1.0, "doc2": 0.8, "doc3": 0.6}],
            targets=[["doc1", "doc3"]],
        )
        metric.update(inputs)
        result = metric.compute()
        # AP = (1/1 + 2/3) / 2 = (1 + 2/3) / 2 = 5/6 H 0.833
        assert abs(result["mean_average_precision"] - 5 / 6) < 1e-6

    def test_mean_average_precision_multiple_queries(self) -> None:
        """Test MAP with multiple queries."""
        metric = MeanAveragePrecision()
        inputs = RankingInput(
            predictions=[
                {"doc1": 1.0, "doc2": 0.8},
                {"doc3": 1.0, "doc4": 0.8},
            ],
            targets=[["doc1"], ["doc3"]],
        )
        metric.update(inputs)
        result = metric.compute()
        # AP for query 1 = 1/1 = 1.0
        # AP for query 2 = 1/1 = 1.0
        # MAP = (1.0 + 1.0) / 2 = 1.0
        assert result == {"mean_average_precision": 1.0}

    def test_ndcg_perfect(self) -> None:
        """Test NDCG with perfect ranking."""
        metric = NDCG(k=3)
        inputs = RankingInput(
            predictions=[{"doc1": 1.0, "doc2": 0.8, "doc3": 0.6}],
            targets=[["doc1", "doc2", "doc3"]],
        )
        metric.update(inputs)
        result = metric.compute()
        assert result == {"ndcg": 1.0}

    def test_ndcg_partial(self) -> None:
        """Test NDCG with non-perfect ranking."""
        metric = NDCG(k=3)
        inputs = RankingInput(
            predictions=[{"doc1": 1.0, "doc2": 0.8, "doc3": 0.6, "doc4": 0.4}],
            targets=[["doc3", "doc4"]],
        )
        metric.update(inputs)
        result = metric.compute()
        # With relevant docs ranked 3rd and 4th, NDCG should be less than 1
        assert 0.0 < result["ndcg"] < 1.0

    def test_ndcg_with_k_limit(self) -> None:
        """Test NDCG with k limit."""
        metric = NDCG(k=2)
        inputs = RankingInput(
            predictions=[{"doc1": 1.0, "doc2": 0.8, "doc3": 0.6, "doc4": 0.4}],
            targets=[["doc1", "doc2", "doc3", "doc4"]],
        )
        metric.update(inputs)
        result = metric.compute()
        # Only considers top 2 documents
        assert result["ndcg"] == 1.0

    def test_ranking_reset(self) -> None:
        """Test reset functionality."""
        metric = MeanAveragePrecision()
        metric.update(
            RankingInput(
                predictions=[{"doc1": 1.0, "doc2": 0.8}],
                targets=[["doc2"]],
            )
        )
        metric.reset()
        metric.update(
            RankingInput(
                predictions=[{"doc1": 1.0, "doc2": 0.8}],
                targets=[["doc1"]],
            )
        )
        result = metric.compute()
        assert result == {"mean_average_precision": 1.0}
