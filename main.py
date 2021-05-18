import data_preprocess
import text_summarization
import pegasus_augmentation
import synonym_rep_augmentation
import transfer_learning
# import predict
from config import DataProcessorConfig


def run_all() -> None:
    var = DataProcessorConfig
    print('+' * 23)
    print('+' * 3 + 'BEGIN PROCESS' + '+' * 4)
    print('+' * 23)
    print()
    print('+' * 5 + 'DATA PRE-PROCESSING' + '+' * 5)
    data = data_preprocess.run_data_preprocessing(var.TEXT_COL)
    print('PROCESS COMPLETED')
    print()
    print('+' * 5 + 'TEXTS SUMMARIZATION' + '+' * 5)
    preprocessed = text_summarization.run_text_summarization(data)
    print('PROCESS COMPLETED')
    print()
    print('+' * 5 + 'PEGASUS AUGMENTATION' + '+' * 5)
    pegasus_data = pegasus_augmentation.execute_pegasus_augmentation(preprocessed)
    print('PROCESS COMPLETED')
    print()
    print('+' * 5 + 'SYNONYM AUGMENTATION' + '+' * 5)
    generated_path = synonym_rep_augmentation.execute_synonym_replacement(pegasus_data)
    print('PROCESS COMPLETED')
    print()
    print('+' * 7 + 'MODEL TRAINING' + '+' * 7)
    transfer_learning.train_model(generated_path)
    print('PROCESS COMPLETED')
    print()

if __name__ == '__main__':
    run_all()