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
This step make sure the dependencies in utils work, **otherwise** you can set your project root to folder generation instead of SynthView.
In this way you will need to modify the following command line as well.

# install pytorch
!!!not yet finished

# Training
upload your target to images for example balloons.png
```commandline
python .\generation\main.py --root .\images\balloons.png
```
python3 main.py --root <path-to-image>
* \<path-to-image\>

# Evaluating
test your result
!!!not yet finished
```commandline
python .\generation\main.py --root .\images\balloons.png --evaluation --model-to-load <path-to-model-pt> --amps-to-load <path-to-amp-pt> --num-steps <number-of-samples> --batch-size <number-of-images-in-batch>
```
python3 main.py --root <path-to-image> --evaluation --model-to-load <path-to-model-pt> --amps-to-load <path-to-amp-pt> --num-steps <number-of-samples> --batch-size <number-of-images-in-batch>
* \<path-to-image\>
* \<path-to-model-pt\>
* \<path-to-amp-pt\>
* \<number-of-samples\>
* \<number-of-images-in-batch\>

# Check generated picture
!!!not yet finished
