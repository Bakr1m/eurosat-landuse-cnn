"""Training orchestration: data -> generators -> fit -> evaluate -> save.

Reproduces the thesis run: up to 50 epochs, ModelCheckpoint on
val_accuracy, EarlyStopping on val_loss (patience 10, restore best).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data import (
    build_dataframe,
    download_eurosat,
    extract_eurosat,
    stratified_split,
)
from src.model import create_cnn_model
from src.preprocessing import (
    flow_from_df,
    make_eval_datagen,
    make_train_datagen,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "eurosat_cnn_best.keras"
EPOCHS = 50


def train(epochs: int = EPOCHS, model_path: Path = BEST_MODEL_PATH):
    archive = download_eurosat()
    image_root = extract_eurosat(archive)
    df = build_dataframe(image_root)
    train_df, val_df, test_df = stratified_split(df)
    print(
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
        f"classes={sorted(df['label'].unique())}"
    )

    train_gen = flow_from_df(make_train_datagen(), train_df, shuffle=True)
    val_gen = flow_from_df(make_eval_datagen(), val_df, shuffle=False)
    test_gen = flow_from_df(make_eval_datagen(), test_df, shuffle=False)

    model = create_cnn_model(num_classes=len(sorted(df["label"].unique())))
    model.summary()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[
            ModelCheckpoint(str(model_path), monitor="val_accuracy",
                            save_best_only=True, mode="max", verbose=1),
            EarlyStopping(monitor="val_loss", patience=10, mode="min",
                          restore_best_weights=True, verbose=1),
        ],
        verbose=2,
    )

    loss, accuracy = model.evaluate(test_gen, verbose=1)
    print(f"Test loss={loss:.4f} accuracy={accuracy:.4f}")

    test_gen.reset()
    steps = test_gen.n // test_gen.batch_size + 1
    y_pred = model.predict(test_gen, steps=steps, verbose=1).argmax(axis=1)
    y_true = test_gen.classes[: len(y_pred)]
    print(classification_report(y_true, y_pred, target_names=test_gen.class_indices))

    model.save(model_path)
    print(f"saved {model_path}")
    return {"test_loss": float(loss), "test_accuracy": float(accuracy)}


if __name__ == "__main__":
    train()
