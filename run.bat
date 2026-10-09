echo 1. TRAINING BASELINE DCGAN
python mnist_train.py

if errorlevel 1 (
    echo ERROR: Baseline training failed.
    goto :error
)

echo 2. MEASURING BASELINE FORGETTING
python measure_forgetting.py ^
    --type baseline

if errorlevel 1 (
    echo ERROR: Baseline forgetting evaluation failed.
    goto :error
)
echo 4. TRAINING REPLAY DCGAN
python mnist_train_replay.py

if errorlevel 1 (
    echo ERROR: Replay training failed.
    goto :error
)

echo 5. MEASURING REPLAY FORGETTING
python measure_forgetting.py ^
    --type replay

if errorlevel 1 (
    echo ERROR: Replay forgetting evaluation failed.
    goto :error
)

