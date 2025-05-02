# NeuroLoop

_Closed-Loop Deep-Brain Stimulation for Controlling Synchronization of Spiking Neurons._

## Running the program

```sh
# Fetch and initialize latest versions of all submodules (ie. ftsts, dbsenv).
$ git submodule update --init --recursive

# Install dependencies, eg.
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install ./dbsenv  # install the dbsenv submodule

# Run the example program.
python src/main.py

# Visualize training results in TensorBoard.
tensorboard --logdir ./ppo_tensorboard  # open http://localhost:6006/
```
