### Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Running the Experiments

###  Training and results 
The complete experiment can be run using the provided batch script

```bash
.\run.bat
```
#### The script performs these steps
1. Trains the baseline DCGAN
2. Saves generator and discriminator checkpoints during training.
3. Measures catastrophic forgetting using the saved baseline checkpoints.
4. Trains the replay-based DCGAN.
5. Saves replay generator and discriminator checkpoints during training.
6. Measures catastrophic forgetting for the replay model.

### Visualize generated sample

Run:

```bash
python visualize_generated_samples.py
```