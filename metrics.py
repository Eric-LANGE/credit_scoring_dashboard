# metrics.py

from typing import Dict, Callable
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    log_loss,
    average_precision_score,
    roc_auc_score,
    make_scorer,
)

# Importer la configuration pour accéder aux constantes comme FN_WEIGHT
from config import Config

# =============================================================================
# Metric Functions
# =============================================================================


def custom_cost_function(
    y_true, y_pred, fn_weight=Config.FN_WEIGHT, sample_weight=None
) -> float:
    """
    Calculates a custom cost that penalizes false negatives more heavily.
    Returns a "score" = 1 - cost, so higher is better, best possible score is 1.
    """
    _, fp, fn, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = (fp + (fn * fn_weight)) / len(y_true)
    return 1 - cost


def confusion_metric(y_true, y_pred, metric) -> float:
    """Generic function to compute one of: tn, fp, fn, tp from the confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics_map = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return float(metrics_map[metric])


def prfs_metric(y_true, y_pred, metric, label, beta=2) -> float:
    """Generic function to compute precision, recall, fscore, or support for a given class label."""
    precision, recall, fscore, support = precision_recall_fscore_support(
        y_true, y_pred, beta=beta
    )
    metric_map = {
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "support": support,
    }
    if metric in metric_map:
        return float(metric_map[metric][label])
    else:
        raise ValueError(f"Unknown metric: {metric}")


# --- Specific metric functions ---
def tn_metric(y_true, y_pred):
    return confusion_metric(y_true, y_pred, "tn")


def fp_metric(y_true, y_pred):
    return confusion_metric(y_true, y_pred, "fp")


def fn_metric(y_true, y_pred):
    return confusion_metric(y_true, y_pred, "fn")


def tp_metric(y_true, y_pred):
    return confusion_metric(y_true, y_pred, "tp")


def precision_metric_n(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "precision", label=0)


def precision_metric_p(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "precision", label=1)


def recall_metric_n(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "recall", label=0)


def recall_metric_p(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "recall", label=1)


def f2_metric_n(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "fscore", label=0)


def f2_metric_p(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "fscore", label=1)


def support_metric_n(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "support", label=0)


def support_metric_p(y_true, y_pred):
    return prfs_metric(y_true, y_pred, "support", label=1)


def pr_auc_metric_n(y_true, y_pred_proba) -> float:
    return average_precision_score(y_true, y_pred_proba, pos_label=0)


def pr_auc_metric_p(y_true, y_pred_proba) -> float:
    return average_precision_score(y_true, y_pred_proba, pos_label=1)


def roc_auc_metric(y_true, y_pred_proba) -> float:
    return roc_auc_score(y_true, y_pred_proba)


# =============================================================================
# Scoring Dictionary
# =============================================================================


def get_scoring_dict() -> Dict[str, Callable]:
    """Create a complete scoring dictionary for model evaluation."""
    # Note: 'response_method' est l'API moderne (>=1.2) et préférée à 'needs_proba'.
    # 'predict' est la valeur par défaut pour les métriques basées sur les classes.
    return {
        # Methods using standard predictions
        "tn": make_scorer(tn_metric, response_method="predict"),
        "fp": make_scorer(fp_metric, response_method="predict"),
        "fn": make_scorer(fn_metric, response_method="predict"),
        "tp": make_scorer(tp_metric, response_method="predict"),
        "precision_n": make_scorer(precision_metric_n, response_method="predict"),
        "precision_p": make_scorer(precision_metric_p, response_method="predict"),
        "recall_n": make_scorer(recall_metric_n, response_method="predict"),
        "recall_p": make_scorer(recall_metric_p, response_method="predict"),
        "f2_score_n": make_scorer(f2_metric_n, response_method="predict"),
        "f2_score_p": make_scorer(f2_metric_p, response_method="predict"),
        "support_n": make_scorer(support_metric_n, response_method="predict"),
        "support_p": make_scorer(support_metric_p, response_method="predict"),
        "custom_cost": make_scorer(custom_cost_function, response_method="predict"),
        # Methods requiring probabilities
        "pr_auc_n": make_scorer(pr_auc_metric_n, response_method="predict_proba"),
        "pr_auc_p": make_scorer(pr_auc_metric_p, response_method="predict_proba"),
        "roc_auc": make_scorer(roc_auc_metric, response_method="predict_proba"),
        "logloss": make_scorer(
            log_loss, response_method="predict_proba", greater_is_better=False
        ),
    }
