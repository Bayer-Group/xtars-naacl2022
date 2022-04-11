import pytest

import xtars.xtars_utils

mixin_utils = xtars.xtars_utils.MixinXTARS()


@pytest.fixture
def load_sample_data_for_test():
    corpus, label_dictionary = mixin_utils.load_sample_data(data_folder="../sample_data", label_col="label",
                                                            label_type="class",
                                                            num_negative_labels_to_sample=3)

    return corpus, label_dictionary


@pytest.fixture
def get_tmp_demo_folder():
    return mixin_utils.get_demo_folder()
