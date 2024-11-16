import io
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def upload_file(
    api: HfApi,
    df: pd.DataFrame,
    path_in_repo: str,
    repo_id: str,
    repo_type: str = "dataset",
    columns: Optional[list[str]] = None,
) -> None:
    file_buffer = io.BytesIO()
    if columns is None:
        df.to_parquet(file_buffer)
    else:
        df[columns].to_parquet(file_buffer)
    file_buffer.seek(0)
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=file_buffer,
        path_in_repo=path_in_repo,
        repo_type=repo_type,
    )


def create_and_push_dataset(
    df: pd.DataFrame, repo_id: str, hf_token: str, config_name: str
) -> None:
    try:
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(df[df["split"] == "train"]),
                "test": Dataset.from_pandas(df[df["split"] == "test"]),
            }
        )
    except KeyError:
        raise KeyError("The DataFrame must contain a 'split' column")

    dataset.push_to_hub(repo_id=repo_id, token=hf_token, config_name=config_name)
