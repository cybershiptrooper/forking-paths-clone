# 🌳 Road not taken

This is the repository for the road not taken project!

## Package management

The project uses `uv` to manage packages. First, install `uv` with `pip install uv`. Then, sync packages by running:
```
uv sync
```

This will create a virtual environment stored in `.venv`.

Next, activate your virtual environment by running:
```
source .venv/bin/activate
```

If you're using windows, launch with
```
.venv/Scripts/activate.exe
```

Alternatively, use `uv run ...` to launch scripts within your virtual environment. **This is done by default** by the bash scripts in the `scripts` folder.

## Launching forking paths

To launch forking paths, we first collect base paths & sort datapoints by uncertainty (how often the base answer matches the final answer of 10 randomly sampled completions).

**Note**: before running data collection, make sure you have huggingface access to the dataset (e.g. [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa)) and models (currently this includes the specified model and [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)).


### Step 1: data collection

Data collection samples 1 base path (greedy decoding) and 10 random paths to estimate the model's uncertainty.

To launch data collection, run the script corresponding to your model in `scripts/data_collection/`:

```
bash scripts/data_collection/{model_name}.sh
```

The results will be saved to `save_locations.collection_folder` in `config.json`.

### Step 2: forking paths

Forking paths takes a single datapoint and re-samples forking rollouts at each sentence chunk.

To launch forking paths, run the script corresponding to your model in `scipts/forking_paths`:

```
scripts/forking_paths/{model_name}.sh
```

The script will read data from `save_locations.selection_folder` and save the results to `save_locations.forking_paths_folder` in `config.json`.

## Adding a model

To add a model, first create an entry within `MODEL_METADATA` in `utils/utils.py`:
```python
MODEL_METADATA = [
    ...,
    "{huggingface_model_path}": {
        "nickname" : "{model_nickname}",
        "reasoning" : {True | False}, # true if trained to reason
    },
]
```

Then, create a `{model_name}.sh` bash file in `scripts/data_collection` and `scripts/forking_paths`. The model nickname will be used as a shorthand when saving model outputs.

## Adding a dataset

Adding a dataset is a longer process. First, add an entry to `data/datasets.json` with information about where to load the dataset (this will be easiest if it's a huggingface dataset). This information is used by the `load_data` function in `data_collection.py` to load the right split.

Next, add a function to parse the right information from the dataset in `utils/data_utils.py`. Right now, only multiple choice questions are supported.

## Shared config

The `config.json` file specifies shared experiment parameters, including save directories and the model used to parse answers.

```java
{
  "save_locations": {
    "dataset_metadata_file": "data/datasets.json", // inputs to data_collection.py
    "collection_folder": "{collection_folder}", // outputs of data_collection.py
    "selection_folder": "{collection_folder}", // inputs to forking_paths.py (ok to reuse collection_folder)
    "forking_paths_folder": "{forking_paths_folder}" // outputs for forking_paths.py (fairly large)
  },
  "experiment_parameters": {
    "answer_model": "{hf_name_or_path}" // model used to parse final answer
  }
}
```

## Streamlit

This repository uses streamlit to quickly view results. To run streamlit locally:
```
streamlit run streamlit_app.py
```
