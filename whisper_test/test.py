from datasets import get_dataset_config_names

dataset_name = "mozilla-foundation/common_voice_12_0"
available_languages = get_dataset_config_names(dataset_name)

print("지원되는 언어 목록:", available_languages)