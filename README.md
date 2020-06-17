# Scane4Me

> Scane4Me is an open-source project for a smart wearable hardware module based on a RaspberryPi, that aims to help blind and visually impared people gain insights from the surrounding environment by just listening to it.

In this repository, you will find:
- `features`: a folder containing Jupyter Notebooks of individual features present in the module
- `images`: contains illustrations of the individual features
- `documentation.md`: a list of all documentations, resources and blogs used
- `src`: source code of the application

Any contribution is more than welcome. If you work with visually impaired people, we would be more than happy to hear from you, build new features based on your feedback and deliver our hardware to your institution. Please reach out to me by [email](mailito:mael.fabien@epfl.ch).

## Getting started

1. Install requirements:

```bash
pip install -r src/requirements.txt
```

2. Run the inference pipeline:

```bash
python src/cv.py
```

## Features

- Live object Detection using YoloV5
- Angle estimation between camera and object
- Producing sound based on type of object and angle in a 3D manner
- Live OCR (to be integrated)