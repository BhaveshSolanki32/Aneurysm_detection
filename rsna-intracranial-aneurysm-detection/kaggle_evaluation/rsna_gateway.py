"""Gateway for the RSNA Intracranial Aneurysm Detection competition."""

import os
import random
import shutil
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import polars as pl

import kaggle_evaluation.core.templates

# Label columns for the competition, from the preparation script.
LABEL_COLS: list[str] = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]
SUBMISSION_ID_COL: str = 'SeriesInstanceUID'


class RSNAGateway(kaggle_evaluation.core.templates.Gateway):
    """Gateway for the RSNA Intracranial Aneurysm Detection competition.

    This gateway handles iterating through the test set, which consists of
    DICOM series. It yields batches of data, where each batch contains the
    file paths to all DICOM images in a single series.
    """

    def __init__(
        self,
        data_paths: tuple[str, str] | None = None,
        file_share_dir: str | None = None,
    ):
        """Initializes the gateway.

        Args:
            data_paths: A tuple of paths to data files. Expected to be
                        (path_to_test.csv, path_to_test_dicom_directory).
        """
        super().__init__(
            data_paths=data_paths,
            file_share_dir=file_share_dir,
            row_id_column_name=SUBMISSION_ID_COL,
        )
        self.set_response_timeout_seconds(30 * 60)  # 30 minutes per series

    def unpack_data_paths(self) -> None:
        """Unpacks data paths from the initialization.

        This method sets the paths to the test CSV file and the directory
        containing the test DICOM files. It assumes a default Kaggle
        environment path if no paths are provided.
        """
        if not self.data_paths:
            # Default paths for the Kaggle environment.
            base_path = os.path.dirname(__file__)
            self.test_csv_path: str = os.path.join(base_path, 'test.csv')
            self.test_dicom_dir: str = os.path.join(base_path, 'series')

        else:
            self.test_csv_path = self.data_paths[0]
            self.test_dicom_dir = self.data_paths[1]

    def generate_data_batches(
        self,
    ) -> Generator[tuple[tuple[str], str], None, None]:
        """Generates data batches for prediction.

        This generator reads the test metadata, and for each series, it
        yields a directory structure corresponding to that series.

        It assumes a directory structure for DICOM files like:
        <test_dicom_dir>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
        """
        test_df: pl.DataFrame = pl.read_csv(self.test_csv_path)

        random.seed(time.time())

        series_instance_uids: list[str] = list(test_df[SUBMISSION_ID_COL].unique())
        shuffled_series_instance_uids: list[str] = random.sample(
            series_instance_uids, len(series_instance_uids)
        )

        for series_uid in shuffled_series_instance_uids:
            series_dir: Path = Path(self.test_dicom_dir) / series_uid
            dicom_paths: list[str | Path] = [str(p) for p in series_dir.glob('*.dcm')]
            # Share the files for the current batch.
            _ = self.share_files(dicom_paths)
            shared_series_dir = Path(
                self.file_share_dir or '/kaggle/shared'
            ) / series_dir.relative_to(series_dir.anchor)
            yield (str(shared_series_dir),), series_uid

    def competition_specific_validation(
        self, prediction: Any, row_id: str, data_batch: Any
    ) -> None:
        """Performs competition-specific validation on a prediction.

        Args:
            prediction: The prediction made by the user's model.
            row_id: The row ID for the prediction (SeriesInstanceUID).
            data_batch: The data batch that was used to generate the prediction.
        """
        pass


if __name__ == '__main__':
    # This block is executed when the script is run directly.
    # In a Kaggle competition, the KAGGLE_IS_COMPETITION_RERUN environment
    # variable will be set.
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        gateway = RSNAGateway()
        # The run() method starts the prediction loop.
        gateway.run()
    else:
        # This allows for local testing without the full Kaggle environment.
        print('Skipping gateway run: Not in a Kaggle competition environment.')
        # Example of how to run locally for testing:
        # 1. Create dummy data:
        #    - A CSV file for test_csv_path
        #    - A directory with dummy DICOM files for test_dicom_dir
        # 2. Set up data_paths to point to the dummy data.
        # 3. Instantiate and run the gateway.
        #
        # Example:
        # dummy_test_csv = 'test.csv'
        # dummy_dicom_dir = 'test_dicoms/'
        # # (code to create dummy files)
        # gateway = RSNAAneurysmGateway(data_paths=(dummy_test_csv, dummy_dicom_dir))
        # gateway.run() # This would need a dummy predict() function.
