import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow_addons as tfa

import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import numpy as np
import pandas as pd
import optuna
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def calc_metrics(obs, pred):
    usecols = ["ACC","PAG","POD","f1_score","tn","fp","fn","tp"]
    obs = obs.round()
    pred = pred.round()
    tn, fp, fn, tp = confusion_matrix(obs, pred).flatten()
    # get percentage
    POFD = fp / (tn + fp) * 100 

    # generate metrics dictionary
    performance = dict(
        ACC=accuracy_score(obs, pred) * 100,
        PAG = precision_score(obs, pred, average='macro') * 100,
        POD=recall_score(obs, pred, average='macro') * 100,
        f1_score=f1_score(obs, pred, average='macro') * 100,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )

    performance = {k: int(v) if k in ['tn','fp','fn','tp'] else float(v) \
                                        for k,v in performance.items()}
    # return metrics with appropriate name
    return dict(zip(usecols, map(performance.get, usecols)))



def soft_f1_loss(y_true, y_pred):
    """
    F1-score의 미분 가능한 버전을 손실 함수로 사용
    y_true: 실제 라벨 (one-hot 인코딩 형식)
    y_pred: 예측 값 (로지스틱 확률 출력)
    """
    # 실제값과 예측값을 float로 변환 (연속적인 확률 값)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # True Positive, False Positive, False Negative 계산
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
    
    # F1-score의 분모와 분자 계산
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # F1-score를 손실로 변환 (1 - F1)
    return 1 - tf.reduce_mean(f1)


def create_sliding_window(data, target, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        # data를 NumPy 배열로 변환한 뒤 flatten
        X.append(data[i:i+window_size].flatten())
        y.append(target[i+window_size])  # 현재 시점(target) 값
    return np.array(X), np.array(y)

def load_data():
    """ Define 3fold evaluation split points using StratifiedGroupKFold

    Args:
        n_splits (int, Optional): Number of splits.
            Default: 3
    """
    pred_hour = 1
    # source = Path(get_original_cwd()) / cfg.catalogue.processed
    # source = '/home/leonid/share/KHOA_seafog_observatory/data/2024/SF_0006_newTd_FC1.pkl'
    source = f'/home/leonid/share/fog_DL/data/2024/processed_agumentation_0906/1h/SF_0006_newTd_FC1.pkl'
    print('data path:',source)

    X = joblib.load(source)["x"]
    # X = X.set_axis([f"{c1}_{c2}" for c1, c2 in X.columns], axis=1)
    y = joblib.load(source)["y"].reset_index()
    y = y.set_index("datetime")
    X = X.join(y[["time_day_sin", "time_day_cos", "time_year_sin", "time_year_cos"]])
    # X = X.join(pd.concat([y.filter(like="time")], axis=1, keys=['time']))

    columns_to_exclude = [f'lag_{str(i).zfill(2)}_label' for i in range(1,7)] #5-3은 라벨피쳐안쓴거
    X = X.drop(columns=columns_to_exclude)
    y = y[f'y_{pred_hour}'].loc[y.index.isin(X.index)]

    drop_mask = X.isna().any(axis=1) | y.isna()
    X = X.iloc[np.flatnonzero(~drop_mask)]  # 46747여개 떨어져나감
    y = y[~drop_mask]
    assert not X.isna().sum().sum()    

    # cv = Custom_CV_3fold()
    return X, y



def objective(trial):

    tf.keras.backend.clear_session()

    n_hidden_1 = trial.suggest_categorical('num_hidden_1', [128, 256])
    n_hidden_2 = trial.suggest_categorical('num_hidden_2', [32, 64, 128])
    n_hidden_3 = trial.suggest_categorical('num_hidden_3', [16, 32, 64])
    
    dropout_rate_1 = trial.suggest_uniform('dropout_rate_1', 0.1, 0.5)
    dropout_rate_2 = trial.suggest_uniform('dropout_rate_2', 0.1, 0.5)
    dropout_rate_3 = trial.suggest_uniform('dropout_rate_3', 0.1, 0.5)
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)    
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])

    model = Sequential([
        Dense(n_hidden_1, activation = 'relu', input_shape = (input_data_shape,)),
        Dropout(dropout_rate_1),
        Dense(n_hidden_2, activation = 'relu'),
        Dropout(dropout_rate_2),
        Dense(n_hidden_3, activation = 'relu'),
        Dropout(dropout_rate_3),
        Dense(2, activation = 'softmax')
    ])
    
    model.compile(optimizer = 'adam', loss = soft_f1_loss, 
                    metrics=[tf.keras.metrics.BinaryAccuracy(), 
                    tfa.metrics.F1Score(num_classes=2, average='macro')])

    history = model.fit(X_train_windowed, y_train_windowed_onehot,
                        validation_data = (X_val_windowed, y_val_windowed_onehot),
                        epochs = 1000,
                        batch_size = batch_size,
                        class_weight = sample_weights,
                        verbose = 1,
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience = 5, mode='max', restore_best_weights = True)])


    val_f1 = max(history.history['val_f1_score'])

    pred = model.predict(X_test_windowed)
    pred = np.argmax(pred, axis = 1)
    obs = np.argmax(y_test_windowed_onehot, axis = 1)

    test_metric = calc_metrics(obs, pred)
    test_f1 = test_metric['f1_score']

    keys_to_extract = ['ACC', 'PAG', 'POD', 'f1_score']
    test_metric2 =[test_metric[key] for key in keys_to_extract] 
    

    filename = f'/home/leonid/share/fog_DL/My_dnn/result/SF_0006_hyperparameter_tuning_results_1h.csv'
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'trial_number', 
                'num_hidden_1',
                'num_hidden_2', 
                'num_hidden_3', 
                'dropout_rate_1', 
                'dropout_rate_2', 
                'dropout_rate_3', 
                'learning_rate', 
                'batch_size',
                'ACC',
                'PAG',
                'POD',
                'F1'
            ])

    # 각 trial의 결과를 파일에 추가한다.
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
        trial.number, 
        trial.params['num_hidden_1'],
        trial.params['num_hidden_2'], 
        trial.params['num_hidden_3'], 
        trial.params['dropout_rate_1'], 
        trial.params['dropout_rate_2'], 
        trial.params['dropout_rate_3'], 
        trial.params['learning_rate'], 
        trial.params['batch_size']
    ] + test_metric2)

    return test_f1


X,y = load_data()

X_train = X.loc[X.index < '2022']
y_train = y.loc[X.index < '2022']

X_valid = X.loc[(X.index >= '2022') & (X.index < '2023')]
y_valid = y.loc[(y.index >= '2022') & (y.index < '2023')]

X_test = X.loc[X.index >= '2023']
y_test = y.loc[y.index >= '2023']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


window_size = 6
X_train_windowed, y_train_windowed = create_sliding_window(X_train, y_train, window_size)
X_val_windowed, y_val_windowed = create_sliding_window(X_valid, y_valid, window_size)
X_test_windowed, y_test_windowed = create_sliding_window(X_test, y_test, window_size)


from sklearn.utils.class_weight import compute_class_weight

# 클래스 가중치 계산
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_windowed),
    y=y_train_windowed
)
sample_weights = dict(enumerate(class_weights))  # 딕셔너리 형태로 변환



from tensorflow.keras.utils import to_categorical

# 타겟 데이터를 One-Hot Encoding으로 변환
y_train_windowed_onehot = to_categorical(y_train_windowed, num_classes=2)
y_val_windowed_onehot = to_categorical(y_val_windowed, num_classes=2)
y_test_windowed_onehot = to_categorical(y_test_windowed, num_classes=2)

input_data_shape = X_train_windowed.shape[1]

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 50)
print(f'Best trial: {study.best_trial.params}')

