from enum import Enum
from typing import Tuple, Optional

bugsinpy_bugs_fail_ground_test = [
    '1', '12', '13', '14', '15', '40', '55', '56', '57', '58', '59', '62', '63', '64', '65'
]

bugsinpy_bugs_all_51_ids = [
    '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
    '26', '27', '28', '29', '30', '31', '32', '33', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47',
    '48', '49', '50', '51', '52', '53', '54', '60', '61', '66', '67', '68'
]

defects4j_bugs_all_255_ids = [str(i) for i in range(1, 256)]

# Backward-compatible alias for code that still imports the old name.
defects4j_bugs_all_116_ids = defects4j_bugs_all_255_ids


def initialize_result_dict(dataset_name):
    neucleus_passed = {}
    neucleus_passed_array = {}
    if dataset_name == 'bugsinpy':
        for bug_id_ in bugsinpy_bugs_all_51_ids:
            neucleus_passed[bug_id_] = 0
            neucleus_passed_array[bug_id_] = []
    elif dataset_name == 'defects4j':
        for bug_id_ in defects4j_bugs_all_255_ids:
            neucleus_passed[bug_id_] = 0
            neucleus_passed_array[bug_id_] = []
    else:
        print(f"invalid dataset_name: {dataset_name}")
        return None, None
    return neucleus_passed, neucleus_passed_array


class HistoryCategory(Enum):
    hafix_agg = '0'

    baseline = '1'

    # blame commit
    baseline_co_evolved_functions_name_modified_file_blame = '2'
    baseline_co_evolved_functions_name_all_files_blame = '3'  # 1

    baseline_all_functions_name_modified_file_blame = '4'
    baseline_all_functions_name_all_files_blame = '5'  # 2

    baseline_all_co_evolved_files_name_blame = '6'

    baseline_function_code_pair_blame = '7'
    baseline_file_code_patch_blame = '8'  # 3

    @property
    def short_name(self):
        return {
            HistoryCategory.hafix_agg: 'HAFix-Agg',
            HistoryCategory.baseline: 'Baseline',
            HistoryCategory.baseline_co_evolved_functions_name_modified_file_blame: 'CFN-modified',
            HistoryCategory.baseline_co_evolved_functions_name_all_files_blame: 'CFN-all',
            HistoryCategory.baseline_all_functions_name_modified_file_blame: 'FN-modified',
            HistoryCategory.baseline_all_functions_name_all_files_blame: 'FN-all',
            HistoryCategory.baseline_all_co_evolved_files_name_blame: 'FLN-all',
            HistoryCategory.baseline_function_code_pair_blame: 'FN-pair',
            HistoryCategory.baseline_file_code_patch_blame: 'FL-diff',
        }[self]

    @classmethod
    def from_setting_key(cls, setting_key: str):
        if setting_key.startswith('setting_'):
            index = setting_key.split('_')[1]
            for member in cls:
                if member.value == index:
                    return member
        raise ValueError(f"Unknown setting key: {setting_key}")


class ModelCategory(Enum):
    codellama_7b = 'codellama_7b'
    deepseek_coder = 'deepseek_coder_6.7b'
    deepseek_coder_v2 = 'deepseek_coder_v2'

    @property
    def name_in_path(self):
        return {
            ModelCategory.codellama_7b: 'codellama_7b_instruct_fp16',
            ModelCategory.deepseek_coder: 'deepseek_coder_6.7b_instruct_fp16',
            ModelCategory.deepseek_coder_v2: 'deepseek_coder_v2_16b_lite_instruct_fp16'
        }[self]

    @property
    def official_name(self):
        return {
            ModelCategory.codellama_7b: 'CodeLlama-Instruct-7B',
            ModelCategory.deepseek_coder: 'DeepSeek-Coder-Instruct-6.7B',
            ModelCategory.deepseek_coder_v2: 'DeepSeek-Coder-V2-Lite-Instruct-16B'
        }[self]

    @property
    def tokenizer_name(self):
        return {
            ModelCategory.codellama_7b: 'codellama/CodeLlama-7b-Instruct-hf',
            ModelCategory.deepseek_coder: 'deepseek-ai/deepseek-coder-6.7b-instruct',
            ModelCategory.deepseek_coder_v2: 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
        }[self]


class PromptCategory(Enum):
    instruction = 'Instruction'
    label = 'InstructionLabel'
    mask = 'InstructionMask'


def get_model_and_prompt_enum(evaluation_dir_name: str) -> Tuple[Optional[ModelCategory], Optional[PromptCategory]]:
    for model in ModelCategory:
        if evaluation_dir_name.startswith(model.value):
            for prompt in PromptCategory:
                if evaluation_dir_name.endswith(prompt.value):
                    return model, prompt
    return None, None
