# Animal4D
Code repo for collect and processing Animal4D data


## Installation

```bash
conda env create -f environment.yml
```

You also need to provide you OpenAI API key:
```bash
export OPENAI_API_KEY=your_openai_api_key
```

## Stage 1: Download Videos
```bash
python download_videos.py --config configs/config.yml
```

## Stage 2: Preprocessing
```bash
python preprocess_vide.py --config configs/config.yml
```

## Stage 3: Run Tracking
```bash
python track_animal.py --config configs/config.yml
```

## Stage 4: Postprocessing
```bash
python build_dataset.py --config configs/config.yml
```