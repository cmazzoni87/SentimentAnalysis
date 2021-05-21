import data_preprocess
import text_summarization
import paraphrasing_augmentation
import synonym_rep_augmentation
import transfer_learning
from config import DataProcessorConfig


def run_all() -> None:
    var = DataProcessorConfig
    print('+' * 28)
    print('+' * 8 + 'BEGIN PROCESS' + '+' * 7)
    print('+' * 28)
    print()
    print('+' * 5 + 'DATA PRE-PROCESSING' + '+' * 5)
    data = data_preprocess.run_data_preprocessing(var.TEXT_COL, var.SOURCE_FILE)
    print('PROCESS COMPLETED')
    print()
    print('+' * 5 + 'TEXTS SUMMARIZATION' + '+' * 5)
    preprocessed = text_summarization.run_text_summarization(data)
    print('PROCESS COMPLETED')
    print()
    print('+' * 5 + 'PEGASUS AUGMENTATION' + '+' * 4)
    pegasus_data = paraphrasing_augmentation.execute_pegasus_augmentation(preprocessed)
    print('PROCESS COMPLETED')
    print()
    print('+' * 5 + 'SYNONYM AUGMENTATION' + '+' * 4)
    generated_path = synonym_rep_augmentation.execute_synonym_replacement(pegasus_data)
    print('PROCESS COMPLETED')
    print()
    print('+' * 7 + 'MODEL TRAINING' + '+' * 7)
    transfer_learning.train_model(generated_path)
    print('PROCESS COMPLETED')
    print()
