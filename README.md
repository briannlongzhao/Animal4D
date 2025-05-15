# Animal4D
Code repo for collect and processing Animal4D data


## Installation

Setup conda environment:
```bash
conda env create -f environment.yml
```

Set Python path:

```
export PYTHONPATH=$(pwd)
```

Set OpenAI API key:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

Download Grounded-SAM2 checkpoint:

```shell
cd externals/Grounded_SAM2/checkpoints && bash download_ckpts.sh
cd externals/Grounded_SAM2/gdino_checkpoints && bash download_ckpts.sh
```

Download ViTPose++ checkpoint:

```shell
mkdir externals/ViTPose/ckpt && cd externals/ViTPose/ckpt
wget https://download.cs.stanford.edu/viscam/Animal4D/ckpt/apt36k.pth
```

## Data Collection

### Stage 1: Download Videos

```bash
python scripts/download_video.py --config configs/config.yml
```

### Stage 2: Preprocessing

```bash
python scripts/preprocess_video.py --config configs/config.yml
```

### Stage 3: Run Tracking

```bash
python scripts/track_animal.py --config configs/config.yml
```

### Stage 4: Postprocessing

```bash
python scripts/build_dataset.py --config configs/config.yml
```

## 4D Reconstruction

### Run 3D/4D Fauna Inference

Follow [3DAnimals](https://github.com/3DAnimals/3DAnimals/) to setup environment.

Update `test_data_dir` variable with path to dataset directory.

For 3D Fauna:

Update `method` variable with "fauna".

```shell
python scripts/reconstruct_video_fauna.py
```

For 4D Fauna:

Update `method` variable with "fauna++".

```shell
python scripts/reconstruct_video_fauna.py
```

### Run SMALify Inference

Follow [SMALify](https://github.com/benjiebob/SMALify) to setup environment

Update `data_dir` variable in `scripts/reconstruct_video_smalify.py` with path to dataset directory.

```shell
python scripts/reconstruct_video_smalify.py
```

### Run Benchmark

```
python scripts/run_benchmark.py --data_dir /path/to/dataset/dir
```

