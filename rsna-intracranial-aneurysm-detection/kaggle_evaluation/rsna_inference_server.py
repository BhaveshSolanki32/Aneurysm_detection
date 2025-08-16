import os
import sys

import kaggle_evaluation.core.templates

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import rsna_gateway


class RSNAInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None, file_share_dir=None):
        return rsna_gateway.RSNAGateway(data_paths, file_share_dir=file_share_dir)
