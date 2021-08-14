import logging
import os
import uuid
from datetime import timedelta
from typing import Optional

import fblearner.flow.api as flow
from aiplatform.modelstore.manifold import manifold_utils
from fblearner.flow.api import types
from iopath.common.file_io import PathManager as PathManagerBase
from iopath.fb.manifold import ManifoldPathHandler

MANIFOLD_PREFIX = "manifold://"
FCU_BUCKET = "fast_content_understanding"
MANIFOLD_API_KEY = "fast_content_understanding-key"
MANIFOLD_BUCKET_NAME = "manifold://fast_content_understanding"
MANIFOLD_BUCKET_DIR = "tree/ecg/flow"

RETENTION_DAYS = 90
DEFAULT_RETENTION_PERIOD = timedelta(days=RETENTION_DAYS).total_seconds()

PathManager = PathManagerBase()
PathManager.register_handler(
    ManifoldPathHandler(timeout_sec=1800, max_parallel=32), allow_override=True
)

logger = logging.getLogger(__name__)


class EcgManifoldHelper(object):
    """
    Helper class for reading/writing Ecg data from/to Manifold.

    For huggingface models, they will be downloaded to local folder and loaded from local.
    Training checkpoints and paras will be saved locally and uploaded to: manifold://fast_content_understanding/tree/ecg/${flow_id}/

    local dir:
    ${local_dir}/
    ${local_dir}/checkpoint/${suffix}
    remote dir:
    manifold://fast_content_understanding/tree/ecg/${flow_id}/
    manifold://fast_content_understanding/tree/ecg/${flow_id}/checkpoint/${suffix}/
    """

    def __init__(
        self,
        task: str,
        load_data_dir: str = "manifold://fast_content_understanding/tree/ecg/data-bin/eur_lex",
        load_model_dir: str = "manifold://fast_content_understanding/tree/ecg/model",
        load_checkpoint_dir: str = "",
        bucket: str = FCU_BUCKET,
        bucket_dir: str = MANIFOLD_BUCKET_DIR,
        overwrite: Optional[bool] = True,
        create_random_dir_if_exists: Optional[bool] = True,
    ) -> None:
        self.task = task
        self.load_data_dir = load_data_dir
        self.load_model_dir = load_model_dir
        self.load_checkpoint_dir = load_checkpoint_dir

        self.flow_id = self.get_flow_id()
        self.prefix = os.path.join(MANIFOLD_PREFIX, bucket, bucket_dir)
        self.overwrite = overwrite
        self.flow_dir = os.path.join(self.prefix, self.flow_id)
        # self.flow_session = FlowSession()
        self.flow_dir = self.mkdir(self.flow_dir, create_random_dir_if_exists)
        self.load_model_local_dir = None
        self.load_checkpoint_local_dir = None
        self.local_dir = self.mkdir(
            os.path.join(os.getcwd(), bucket, "tree", "ecg", self.flow_id),
            True,
        )

    def get_flow_id(self):
        run_id = flow.get_flow_environ().workflow_run_id
        # run_id returns 1 for local tests
        flow_id = str(run_id) if (run_id and run_id != 1) else "f" + str(uuid.uuid4())
        logger.info(f"current flow id is {flow_id}")
        return flow_id

    def mkdir(self, dir_name, create_random_dir_if_exists: Optional[bool] = True):

        if not PathManager.exists(dir_name):
            PathManager.mkdirs(dir_name)
            logger.info(f"A new directory: {dir_name} is created.")
        else:
            logger.info(f"Directory: {dir_name} already exists.")
            if create_random_dir_if_exists:
                dir_name = dir_name + "_" + str(uuid.uuid4())
                PathManager.mkdirs(dir_name)
                logger.info(f"A random directory: {dir_name} is created.")

        return dir_name

    def get_resource_dir_local(self, manifold_dir):
        """
        Given a manifold path, download the resource from it to cached local path and return the local path
        must be a path to files/resource, a path to dir won't work

        """
        # dir parameter in PathManager.get_local_path must end with "/"
        assert PathManager.isdir(manifold_dir)
        manifold_dir = manifold_dir.strip("/") + "/"
        local_dir = PathManager.get_local_path(manifold_dir, recursive=True)
        logger.info(f"cached local path is: {local_dir}")
        return local_dir

    def save_checkpoint_manifold(self, suffix: Optional[str] = ""):
        """
        save checkpoint files locally and upload it to manifold:

        pytorch_model.bin
        configs.json
        training_para.pth
        """
        # TODO: create random local path?
        """
        checkpoint data is saved in ${flow_dir}/checkpoint/${suffix}
        eg: manifold://fast_content_understanding/tree/ecg/flow/f123456/checkpoint/epoch050/

        """
        manifold_checkpt_dir = os.path.join(self.flow_dir, "checkpoint", suffix)
        manifold_checkpt_dir = self.mkdir(
            manifold_checkpt_dir, create_random_dir_if_exists=False
        )

        local_checkpt_dir = os.path.join(self.local_dir, "checkpoint", suffix)
        save_files = os.listdir(local_checkpt_dir)

        for save_file in save_files:
            local_file_path = os.path.join(local_checkpt_dir, save_file)
            manifold_file_path = os.path.join(manifold_checkpt_dir, save_file)
            PathManager.copy_from_local(
                local_file_path, manifold_file_path, overwrite=True
            )

        # TODO: clear the space if too many files

    def save_training_result_manifold(self, file_lst):
        """
        training record files are saved in ${flow_dir}/
        locally, they are in ${local_dir}/

        output_dev
        output_train
        predictions_dev

        """
        for save_file in file_lst:
            local_file_path = os.path.join(self.local_dir, save_file)
            if PathManager.isfile(local_file_path):
                manifold_file_path = os.path.join(self.flow_dir, save_file)
                PathManager.copy_from_local(
                    local_file_path, manifold_file_path, overwrite=True
                )
        logger.info(f"Uploading {file_lst} is completed.")


def gen_random_manifold_path(
    ttl: Optional[int] = DEFAULT_RETENTION_PERIOD,
    bucket: Optional[str] = MANIFOLD_BUCKET_NAME,
    suffix: Optional[str] = None,
    **kwargs,
) -> types.ManifoldPath:
    # The path is ready to be written,
    # but is not ready to be read unless written to it
    if ttl is None:
        ttl = DEFAULT_RETENTION_PERIOD.total_seconds()
    # manifold_dir = get_flow_dir()
    manifold_dir = f"{MANIFOLD_BUCKET_NAME}/{MANIFOLD_BUCKET_DIR}"
    path = os.path.join(manifold_dir, str(uuid.uuid4()))
    if suffix:
        path += suffix
    manifold_path = types.MANIFOLDPATH.new(path=path)
    print(manifold_path)
    manifold_utils.initialize_directory(manifold_dir, ttl, **kwargs)
    return manifold_path


def gen_empty_manifold_path(
    ttl: Optional[int] = DEFAULT_RETENTION_PERIOD,
    suffix: Optional[str] = None,
    **kwargs,
) -> types.ManifoldPath:
    # The path is ready to be read
    manifold_path = gen_random_manifold_path(ttl, suffix=suffix, **kwargs)
    print("manifold_path.uri is : ", manifold_path.uri)
    with PathManager.open(manifold_path.uri, "w", **kwargs) as fd:
        # Write an empty string
        fd.write("")
    return manifold_path
