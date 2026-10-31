"""
Athena 3.0
Paper Trading Model Manager V3

Loads the five selected Athena models from the permanent project
models directory.

Safety:
    - Model loading only.
    - No broker connection.
    - No market-data connection.
    - No order placement.
    - No paper trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import time


# ============================================================================
# PROJECT ROOT
# ============================================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (
            (parent / "models").is_dir()
            and (parent / "data").is_dir()
            and (parent / "src").is_dir()
        ):
            return parent

    raise RuntimeError(
        "Could not locate Athena 3.0 project root."
    )


PROJECT_ROOT = find_project_root()
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================================
# TRANSFORMER CUSTOM LAYERS
# ============================================================================

def get_transformer_custom_objects():
    """
    Recreates the exact custom layers used by Transformer V1.

    These definitions match:
        src/athena/models/transformer/train_transformer_v1.py
    """

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    class PositionalEmbedding(layers.Layer):

        def __init__(
            self,
            sequence_length,
            embedding_dim,
            **kwargs,
        ):
            super().__init__(**kwargs)

            self.sequence_length = sequence_length
            self.embedding_dim = embedding_dim

            self.feature_projection = layers.Dense(
                embedding_dim
            )

            self.position_embedding = layers.Embedding(
                input_dim=sequence_length,
                output_dim=embedding_dim,
            )

        def call(self, inputs):

            positions = tf.range(
                start=0,
                limit=self.sequence_length,
                delta=1,
            )

            projected = self.feature_projection(
                inputs
            )

            positional = self.position_embedding(
                positions
            )

            return projected + positional

        def get_config(self):

            config = super().get_config()

            config.update(
                {
                    "sequence_length": self.sequence_length,
                    "embedding_dim": self.embedding_dim,
                }
            )

            return config


    class TransformerBlock(layers.Layer):

        def __init__(
            self,
            embedding_dim,
            num_heads,
            feed_forward_dim,
            dropout_rate,
            **kwargs,
        ):
            super().__init__(**kwargs)

            self.attention = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads,
                dropout=dropout_rate,
            )

            self.feed_forward = keras.Sequential(
                [
                    layers.Dense(
                        feed_forward_dim,
                        activation="relu",
                    ),
                    layers.Dropout(
                        dropout_rate
                    ),
                    layers.Dense(
                        embedding_dim
                    ),
                ]
            )

            self.layer_norm_1 = layers.LayerNormalization(
                epsilon=1e-6
            )

            self.layer_norm_2 = layers.LayerNormalization(
                epsilon=1e-6
            )

            self.dropout_1 = layers.Dropout(
                dropout_rate
            )

            self.dropout_2 = layers.Dropout(
                dropout_rate
            )

        def call(
            self,
            inputs,
            training=False,
        ):

            normalized_inputs = self.layer_norm_1(
                inputs
            )

            attention_output = self.attention(
                normalized_inputs,
                normalized_inputs,
                training=training,
            )

            attention_output = self.dropout_1(
                attention_output,
                training=training,
            )

            residual_1 = (
                inputs + attention_output
            )

            normalized_residual = self.layer_norm_2(
                residual_1
            )

            feed_forward_output = self.feed_forward(
                normalized_residual,
                training=training,
            )

            feed_forward_output = self.dropout_2(
                feed_forward_output,
                training=training,
            )

            return residual_1 + feed_forward_output

        def get_config(self):

            config = super().get_config()

            config.update(
                {
                    "embedding_dim": self.attention.key_dim
                    * self.attention.num_heads,
                    "num_heads": self.attention.num_heads,
                    "feed_forward_dim": self.feed_forward.layers[0].units,
                    "dropout_rate": self.dropout_1.rate,
                }
            )

            return config


    return {
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerBlock": TransformerBlock,
    }


# ============================================================================
# MODEL SPECIFICATIONS
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    relative_path: Path
    description: str


MODEL_SPECS = {

    "XGBoost": ModelSpec(
        name="XGBoost",
        model_type="xgboost",
        relative_path=Path(
            "models",
            "xgboost",
            "xgboost_v5_early_stopping.json",
        ),
        description="Chronological XGBoost V5.",
    ),

    "LightGBM": ModelSpec(
        name="LightGBM",
        model_type="lightgbm",
        relative_path=Path(
            "models",
            "lightgbm",
            "lightgbm_v4_early_stopping.txt",
        ),
        description="Chronological LightGBM V4.",
    ),

    "CatBoost": ModelSpec(
        name="CatBoost",
        model_type="catboost",
        relative_path=Path(
            "models",
            "catboost",
            "catboost_v4_early_stopping.cbm",
        ),
        description="Chronological CatBoost V4.",
    ),

    "LSTM": ModelSpec(
        name="LSTM",
        model_type="lstm",
        relative_path=Path(
            "models",
            "lstm",
            "lstm_v2_model.keras",
        ),
        description="Validated scaled LSTM V2.",
    ),

    "Transformer": ModelSpec(
        name="Transformer",
        model_type="transformer",
        relative_path=Path(
            "models",
            "transformer",
            "transformer_v1_model.keras",
        ),
        description="Transformer V1 with custom layers.",
    ),
}


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class ModelLoadResult:

    name: str
    model_type: str
    path: str
    loaded: bool
    load_time_seconds: float

    model: Any = None
    error: str | None = None


# ============================================================================
# MODEL LOADERS
# ============================================================================

def _load_xgboost(path: Path) -> Any:

    import xgboost as xgb

    model = xgb.XGBClassifier()

    model.load_model(
        str(path)
    )

    return model


def _load_lightgbm(path: Path) -> Any:

    import lightgbm as lgb

    return lgb.Booster(
        model_file=str(path)
    )


def _load_catboost(path: Path) -> Any:

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()

    model.load_model(
        str(path)
    )

    return model


def _load_lstm(path: Path) -> Any:

    from tensorflow import keras

    return keras.models.load_model(
        str(path),
        compile=False,
    )


def _load_transformer(path: Path) -> Any:

    from tensorflow import keras

    custom_objects = (
        get_transformer_custom_objects()
    )

    return keras.models.load_model(
        str(path),
        custom_objects=custom_objects,
        compile=False,
    )


# ============================================================================
# LOAD ONE MODEL
# ============================================================================

def load_model(
    spec: ModelSpec,
) -> ModelLoadResult:

    path = PROJECT_ROOT / spec.relative_path

    start = time.perf_counter()

    if not path.exists():

        return ModelLoadResult(
            name=spec.name,
            model_type=spec.model_type,
            path=str(path),
            loaded=False,
            load_time_seconds=0.0,
            error=(
                f"Model file does not exist: {path}"
            ),
        )

    try:

        if spec.model_type == "xgboost":

            model = _load_xgboost(path)

        elif spec.model_type == "lightgbm":

            model = _load_lightgbm(path)

        elif spec.model_type == "catboost":

            model = _load_catboost(path)

        elif spec.model_type == "lstm":

            model = _load_lstm(path)

        elif spec.model_type == "transformer":

            model = _load_transformer(path)

        else:

            raise ValueError(
                f"Unsupported model type: "
                f"{spec.model_type}"
            )

        elapsed = (
            time.perf_counter() - start
        )

        return ModelLoadResult(
            name=spec.name,
            model_type=spec.model_type,
            path=str(path),
            loaded=True,
            load_time_seconds=elapsed,
            model=model,
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter() - start
        )

        return ModelLoadResult(
            name=spec.name,
            model_type=spec.model_type,
            path=str(path),
            loaded=False,
            load_time_seconds=elapsed,
            error=(
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================================
# LOAD ALL MODELS
# ============================================================================

def load_all_models():

    results = {}

    for name, spec in MODEL_SPECS.items():

        results[name] = load_model(
            spec
        )

    return results


# ============================================================================
# STREAMLIT CACHED LOADER
# ============================================================================

def load_all_models_cached():

    import streamlit as st

    @st.cache_resource(
        show_spinner=False
    )
    def _cached_loader():

        return load_all_models()

    return _cached_loader()


# ============================================================================
# SUMMARY
# ============================================================================

def summarize_results(results):

    total = len(results)

    loaded = sum(
        1
        for result in results.values()
        if result.loaded
    )

    failed = total - loaded

    total_load_time = sum(
        result.load_time_seconds
        for result in results.values()
    )

    return {
        "project_root": str(
            PROJECT_ROOT
        ),
        "models_directory": str(
            MODELS_DIR
        ),
        "total_models": total,
        "loaded_models": loaded,
        "failed_models": failed,
        "total_load_time_seconds":
            total_load_time,
        "all_loaded": failed == 0,
    }


# ============================================================================
# TEST
# ============================================================================

def main():

    print("=" * 72)
    print(
        "ATHENA 3.0 - MODEL MANAGER V3"
    )
    print("=" * 72)

    print()
    print(
        f"Project root : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Models dir   : "
        f"{MODELS_DIR}"
    )

    print()
    print(
        "Loading all registered models..."
    )
    print()

    results = load_all_models()

    for name, result in results.items():

        status = (
            "READY"
            if result.loaded
            else "FAILED"
        )

        print(
            f"[{status:6}] "
            f"{name:12} "
            f"{result.load_time_seconds:8.3f}s"
        )

        print(
            f"         {result.path}"
        )

        if result.error:

            print(
                f"         ERROR: "
                f"{result.error}"
            )

        print()

    summary = summarize_results(
        results
    )

    print("-" * 72)

    print(
        f"Loaded: "
        f"{summary['loaded_models']}/"
        f"{summary['total_models']}"
    )

    print(
        f"Total load time: "
        f"{summary['total_load_time_seconds']:.3f}s"
    )

    print("-" * 72)

    if summary["all_loaded"]:

        print(
            "MODEL MANAGER STATUS: READY"
        )

        print()
        print(
            "All 5 models loaded successfully."
        )

        return 0

    print(
        "MODEL MANAGER STATUS: FAILED"
    )

    print(
        "At least one required model failed."
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )