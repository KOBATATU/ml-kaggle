import os

"""
作業ディレクトリにディレクトリが戻り、その中で設定したディレクトリを作る。

PROJECT_ROOT:作業ディレクトリに繋ぐパス

DATA_DIR:データが入るディレクトリを作る

FEATURE_DIR:特徴量が入るディレクトリ

PROCESSED_ROOT:実行されたディレクトリ。この中にログを入れる

RAW_DATA_DIR:DATA_DIRのディレクトリの中に使用するデータを入れる

PREDICT_FOR_TEST_DIR:DATA_DIRのディレクトリの中にtestを入れる


"""

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

FEATURE_DIR = os.path.join(DATA_DIR, 'feature')
os.makedirs(FEATURE_DIR, exist_ok=True)

PROCESSED_ROOT = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_ROOT, exist_ok=True)

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

PREDICT_FOR_TEST_DIR = os.path.join(DATA_DIR, 'predict_test')
os.makedirs(PREDICT_FOR_TEST_DIR, exist_ok=True)
