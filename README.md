### Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirement.txt
```

---

### Running the Experiments

### Baseline training

Run the standard DCGAN:

```bash
python mnist_train.py
```
### Replay training

Run:

```bash
python train_replay.py
```
## Evaluating Catastrophic Forgetting

Run:

```bash
python evaluate_forgetting.py
```
