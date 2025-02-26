# SynthView

LU CDS525 Project SynthView Team

Do the following steps

# Clone repository

```commandline
git clone https://github.com/RockyRori/SynthView
cd ./SynthView
```

# Install dependencies

```commandline
python -m pip install -r ./requirements.txt
```

# Set project source

If you use **pycharm** then right click folder generation and choose the down most mark directory as sources root.
This step make sure the dependencies in utils work, **otherwise** you can set your project root to folder generation
instead of SynthView.
In this way you will need to modify the following command line as well.

# prerequisite

## Assure CUDA available

```commandline
nvidia-smi
```

## Sample Result

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 566.03 Driver Version: 566.03 CUDA Version: 12.7 |
|-----------------------------------------+------------------------+----------------------+
| GPU Name Driver-Model | Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr:Usage/Cap | Memory-Usage | GPU-Util Compute M. |
| | | MIG M. |
|=========================================+========================+======================|
| 0 NVIDIA GeForce RTX 4080 ... WDDM | 00000000:01:00.0 Off | N/A |
| N/A 55C P0 31W / 155W | 0MiB / 12282MiB | 0% Default |
| | | N/A |
+-----------------------------------------+------------------------+----------------------+

## uninstall pytorch

check pytorch type

```python
import torch

print(torch.cuda.is_available())
```

If True skip., otherwise do the following

```commandline
pip uninstall torch
```

## install pytorch with cuda

official website to see

https://pytorch.org/

choose appropriate pytorch version and download
![pytorch.png](./figures/pytorch.png)

# Training

upload your target to images for example balloons.png

```commandline
python .\generation\main.py --root .\images\balloons.png
```

python3 main.py --root <path-to-image>

* \<path-to-image\>

run the previous command you will see
![training.png](figures/training.png)

# Evaluating

test your result
!!!not yet finished

```commandline
python .\generation\main.py --root .\images\balloons.png --evaluation --model-to-load .\results\2025-02-26_11-17-13\g_multivanilla.pt --amps-to-load .\results\2025-02-26_11-17-13\amps.pt --num-steps 100 --batch-size 16
```

python3 main.py --root <path-to-image> --evaluation --model-to-load <path-to-model-pt> --amps-to-load <path-to-amp-pt>
--num-steps <number-of-samples> --batch-size <number-of-images-in-batch>

* \<path-to-image\>
* \<path-to-model-pt\>
* \<path-to-amp-pt\>
* \<number-of-samples\>
* \<number-of-images-in-batch\>

run the previous command you will see
![evaluatiom.png](figures/evaluatiom.png)

# Check generated picture

at the same hierarchy you will find a results folder containing everything.
![sampled.png](figures/sampled.png)