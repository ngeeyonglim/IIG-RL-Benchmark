# IIG-RL-Benchmark

IIG-RL-Benchmark is a library for running game theoretical and deep RL algorithms on OpenSpiel games. Furthermore, we compute exact exploitability using the [DH3 library](https://github.com/gabrfarina/dh3) which currently supports Phantom Tic-Tac-Toe and 3x3 Dark Hex, as well as their abrupt versions.

Paper: [TODO](https://arxiv.com)

## Citation

```
TODO
```

## Installation

Ensure that your conda is not activated to prevent spawning dual virtual environments.

```bash
# Download this repository (IIG-RL-Benchmark) and install dependencies
git clone https://github.com/nathanlct/IIG-RL-Benchmark.git
cd IIG-RL-Benchmark

# updates the dh3 repo with approriate commit
git submodule init
git submodule update

# do this from the top level repo! This will install dh3 and some dependencies. Make sure to do this from top level repo.
pixi install
pixi shell

# Install our custom OpenSpiel (with fixes fixes for PTTT and DH3 and the addition of abrupt PTTT) (our default is python = 3.11 )
# If your system isn't linux or you're not on python 3.11, choose the variant for your system (https://github.com/nathanlct/open_spiel/releases)
pip uninstall open_spiel -y
pip install https://github.com/nathanlct/open_spiel/releases/download/v1.pttt/open_spiel-1.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Lastly, install the other requirements
pip install -r requirements.txt
```

### Installing DH3 on MacOS

The DH3 dependency (which is only required to compute exact exploitability of policies) supports installation on Linux. To install on OSX, use the following branch in DH3 (which has not been extensively tested): `osx_install` ([link](https://github.com/gabrfarina/dh3/tree/osx_install)).

### (Optional alternative) Installing DH3 via `pip`

```
cd IIG-RL-Benchmark
pip install ./dh3 
```

## Usage

Train a PPO agent on Phantom Tic-Tac-Toe for 10M steps with an entropy coefficient of 0.05 and no exploitability computation.

```bash
python main.py algorithm=ppo game=classical_phantom_ttt max_steps=10000000 algorithm.ent_coef=0.05 compute_exploitability=False
```

Train an NFSP agent on Abrupt Dark Hex 3 for 10M steps and compute exact exploitability every 1M steps.

```bash
python main.py algorithm=nfsp game=abrupt_dark_hex max_steps=10000000 compute_exploitability=True compute_exploitability_every=1000000
```

The following algorithms are implemented: `ppo`, `nfsp`, `ppg`, `mmd`, `rnad`, `escher`, `psro`.

Exact exploitability is supported for the following games: `classical_phantom_ttt`, `abrupt_phantom_ttt`, `classical_dark_hex`, `abrupt_dark_hex`.

See the `configs` folder for all available command-line arguments.

## Evaluating head2head 


- Input the paths and algorithm names into a `yaml` file like the example in `head2head/example.yaml`


```
python head2head/head2head_eval.py --agents-yaml path/to/example.yaml --save-dir <save-dir>
```

This command will kick off head2head evaluation for the agents listed in the eval `yaml` using the `dh3` library. 


