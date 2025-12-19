import numpy as np
import moabb
import mne

from sklearn.metrics import matthews_corrcoef
from sklearn.base import BaseEstimator, TransformerMixin

import time
import matplotlib.pyplot as plt
import pandas as pd



class PseudoOnlineWindow():
    """
    cria janelas deslizantes e as rotula, baseado no arigo do framework pseudo online
    os rotulos sao dados de acordo com a classe majoritaria da janela

    raw: objeto mne.Raw
    events: array de eventos padrao do mne
    interval: parametro do dataset que define os intervalos de imagetica
    task_ids: define quais os ids das tasks que vao ser classificadas (permite problema multiclasse ou one-versus-rest, por exemplo)
    window_size: define tamanho (em segundos) da janela
    window_step: distancia entre os inicios de duas janelas adjacentes, entao define a sobreposicao entre janelas
    """
    def __init__(self, raw, events, interval, task_ids, window_size, window_step, chan_list=None):
        self.raw = raw
        self.events = events
        self.interval = interval
        self.sfreq = raw.info['sfreq']
        self.task_ids = task_ids

        self.window_size = int(window_size * self.sfreq)
        self.window_step = int(window_step * self.sfreq)
        self.chan_list = chan_list

        self.t_start = int(interval[0] * self.sfreq)
        self.t_end = int(interval[1] * self.sfreq)

        self.labels = self.generate_labels()

    def generate_labels(self):
        """
        atribui uma classe para cada amostra do dado. inicializa o vetor de rotulos em 0 e atribui a classe da task
        às amostras do período de imagética
        """
        
        #aqui n_sample eh de uma forma e la embaixo de outra
        n_samples = self.raw.n_times
        labels = np.zeros(n_samples, dtype=int)

        valid_ids = list(self.task_ids.values()) #vai selecionar so os eventos que queremos

        for ev in self.events:
            ev_idx, _, ev_id = ev

            if ev_id in valid_ids:
                # considera so o periodo de imagetica para rotular como task
                start = ev_idx + self.t_start
                stop = ev_idx + self.t_end

                # garante limites do array
                start = max(0, start)
                stop = min(n_samples, stop)

                labels[start:stop] = ev_id
        return labels


    def generate_windows(self):
        """
        gera janelas para todo o dado, alem de conter a logica de desempate de classe
        return: array de dados das janelas X (shape=(2,)), array de labels y de cada janela, array de tempos em s de inicio de cada janela
        """
        X, y, times = [], [], []

        data = self.raw.get_data()  #(n_channels, n_samples)
        #n_samples ta sendo obtido de outra forma la em cima
        n_samples = data.shape[1]

        for start_idx in range (0, n_samples - self.window_size, self.window_step):
            end_idx = start_idx + self.window_size

            if self.chan_list == None:
                window_data = data[:, start_idx : end_idx]
                window_labels = self.labels[start_idx:end_idx]
            else:       #seleçao de canais
                window_data = []
                for chan in self.chan_list:
                    if chan in self.raw.ch_names:
                        window_data.append(data[chan, start_idx : end_idx])
                    else:
                        raise ValueError(f"Canal {chan} não está na lista {self.raw.ch_names}")
                window_labels = self.labels[start_idx:end_idx]

            count = np.bincount(window_labels)
            major = np.argmax(count)

            prop_major = count[major] / len(window_labels)

            #define a proporcao de empate
            n_classes = len(np.unique(window_labels))
            draw_prop = 1 / n_classes

            if prop_major != draw_prop:
                y.append(major)
            #se ha empate, vence a classe posterior; acho que agora ta tratando de quaisquer qtd de classes na janela
            else:
                y.append(window_labels[-1])

            X.append(window_data)
            times.append(((start_idx / self.sfreq), (end_idx / self.sfreq)))

        #um ponto importante a se checar é se o shape de X vai funcionar bem nos pipelines
        return np.array(X), np.array(y), np.array(times)


class IdleBaseline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.rest_label = 0

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("array de labels está faltando")
        
        idle_windows = (y == self.rest_label)

        if not np.any(idle_windows):
            raise ValueError("não há janelas de rest nos dados de treino")
        
        self.baseline_ = X[idle_windows].mean(axis=0)
        return self

    def transform(self, X):
        return np.subtract(X, self.baseline_)


class ERDS():

    def __init__():
        pass


class PSD():

    def __init__():
        pass







class PseudoOnlineEvaluation():
    """
    faz avaliacao com janelas deslizantes, tanto na mesma sessao quanto inter sessao.

    dataset: dataset utilizado
    pipelines: dict de pipelines
    method: pode ser 'within-session' para avaliacao na mesma sessao ou 'inter-session' para avaliacao entre sessoes
        within session: treina nas primeiras k trials definidas por ratio e testa nas demais, dentro de uma unica sessao
        inter session: treina nas prmeiras k sessoes definidas por ratio e testa nas demais sessoes
    ratio: define a proporçao dos dados usada para treino
    
    """
    def __init__(self, dataset, pipelines, method, wsize, wstep, subjects, ratio=0.7, no_run=False):
        self.dataset = dataset
        self.pipelines = pipelines
        self.ratio = ratio
        self.method = method
        self.wsize = wsize
        self.wstep = wstep
        self.subjects = subjects
        self.no_run = no_run
        
        self.results_ = []

    def raw_concat(self, raw_list):
        """
        essa função garante que nao vai ter erro na concatenação dos raws, pro caso de a lista estar vazia, ter um só elemento, ou for uma lista com vários raws (como se é esperado)
        """
        if len(raw_list) == 0:
            raise ValueError("A lista de raws está vazia")
        elif len(raw_list) == 1:    #aqui, o raw de uma sessao pode ser constituido de 1 ou mais runs, por isso essa verificação
            if type(raw_list[0]) != list:
                return raw_list[0]
            else:
                return mne.concatenate_raws(raw_list[0])
        else:
            return mne.concatenate_raws(raw_list)

    def evaluate(self):
        """
        ******ainda precisa ser validado*******
        ******adicionar verificacoes de erro e de tipos**********
        ******adicionar interpretabilidade e rastreabilidade*******      
        """

        for subject in self.subjects:
            if subject not in self.dataset.subject_list:
                raise ValueError(f"Índice de sujeito inválido: {subject}")
            else:
                print(f"Processando sujeito {subject}...")

                raws_dict = {}
                raws_test = {}
                pre = self.dataset.get_data(subjects=[subject])

                """
                o within session faz split nas janelas, mas talvez o melhor seja fazer split como no inter-session (separar os raws de treino e teste antes de gerar janela)
                de todo modo, isso é tranquilo de mudar
                """
                
                session_keys = []   #armazenar os ids de sessoes (que nem sempre sao ints)

                if self.method == 'within-session':
                    
                    #o raw mesmo fica muito aninhado dentro do dict do get_data, entao tem que acessar uma penca de dicionario ate chegar la
                    for _, runs in pre.items():
                        for sess, dicts in runs.items():
                            session_keys.append(sess)
                            raws_dict[sess] = []
                            for _, data in dicts.items():
                                raws_dict[sess].append(data)
                    
                    for sess in session_keys:
                        print(f"Processando sessão {sess} sujeito {subject}...")
                        raw = self.raw_concat(raws_dict[sess]) #concatena todos os raws e gera o split depois
                        events, event_ids = mne.events_from_annotations(raw)  #aqui da pra extrair o array de eventos pra usar no gerador de janelas

                        wgen = PseudoOnlineWindow(raw=raw,
                                                events=events,
                                                interval=self.dataset.interval,
                                                task_ids=event_ids,
                                                window_size=self.wsize,
                                                window_step=self.wstep
                                                )
                        
                        X, y, times = wgen.generate_windows()   #o times pode ser usado depois pra plot, etc

                        idx_split = int(len(X) * self.ratio)

                        times_test = times[idx_split:]

                        X_train, y_train = X[:idx_split], y[:idx_split]
                        X_test, y_test = X[idx_split:], y[idx_split:]

                        if (self.no_run):
                            return X_train, y_train, X_test, y_test

                        for name, pipe in self.pipelines.items():
                            t_start = time.perf_counter()

                            print("Treinando...")
                            pipe.fit(X_train, y_train)
                            print("Modelo treinado!")

                            t_end = time.perf_counter()
                            t_train = t_end - t_start


                            print(f"Tempo de treino: {t_train}")

                            predictions = []
                            mcc_acc = []

                            for window in range(len(X_test)):
                                window_start = times_test[window][0]
                                window_end = times_test[window][1]

                                t_start = time.perf_counter()

                                print("Realizando previsão...")
                                y_pred = pipe.predict([X_test[window]])[0]
                                print("Predict")

                                t_end = time.perf_counter()

                                predictions.append(y_pred)
                                t_predict = t_end - t_start

                                print(f"Tempo de predição: {t_predict}")

                                mcc_acc = matthews_corrcoef(y_test[:window+1], predictions[:window+1]) #score mcc acumulado até a janela

                                res = {
                                    "dataset": self.dataset,
                                    "subject": subject,
                                    "session": sess,
                                    "method": self.method,
                                    "pipeline": name,
                                    "t_train": t_train,
                                    "window": window,
                                    "window_start": window_start,
                                    "window_end": window_end,
                                    "t_predict": t_predict,
                                    "y_pred": y_pred,
                                    "y_true": y_test[window],
                                    "correct": (predictions[window] == y_test[window]),
                                    "mcc_acc": mcc_acc
                                }
                                
                                self.results_.append(res)

                elif self.method == 'inter-session':
                    if self.dataset.n_sessions > 1:
                        #split de sessoes
                        session_split = int(self.ratio * self.dataset.n_sessions)
                        raws_list = []
                        raws_train = []

                        print(f"O índice de sessões de treino é {session_split}, o dataset possui {self.dataset.n_sessions} sessões por sujeito")

                        for _, runs in pre.items():
                            for sess, dicts in runs.items():
                                session_keys.append(sess)
                                raws_test[sess] = []
                                raws_dict[sess] = []
                                for _, data in dicts.items():   #salva separado os dados de treino e de teste
                                    raws_dict[sess].append(data)
                        
                        #essa verificação é porque eu acahva que o int() arredondava pra cima o valor... de todo jeito, nao faz mal deixar isso aqui
                        if session_split == self.dataset.n_sessions:
                            train_sessions = session_keys[:(session_split - 1)]
                            test_sessions = session_keys[(session_split - 1):]
                            for sess, data in raws_dict.items():
                                if (session_keys.index(sess) + 1) < session_split:
                                    raws_list.append(data)
                                else:
                                    raws_test[sess].append(data)
                        else:
                            train_sessions = session_keys[:session_split]
                            test_sessions = session_keys[session_split:]
                            for sess, data in raws_dict.items():
                                if (session_keys.index(sess) + 1) <= session_split:
                                    raws_list.append(data)
                                else:
                                    raws_test[sess].append(data)
                        
                        
                        raws_train = self.raw_concat(raws_list)
                        print("sessoes de treino concatenadas")

                        events, event_ids = mne.events_from_annotations(raws_train)

                        wgen_train = PseudoOnlineWindow(raw=raws_train,
                                                            events=events,
                                                            interval=self.dataset.interval,
                                                            task_ids=event_ids,
                                                            window_size=self.wsize,
                                                            window_step=self.wstep
                                                            )

                        X_train, y_train, times_train = wgen_train.generate_windows()

                        if(self.no_run):
                            return X_train, y_train
                            
                        print(f"Treinando nas sessões {train_sessions}...")

                        for name, pipe in self.pipelines.items():
                            print(f"Pipeline: {name}")
                            print("Treinando modelo")
                            t_start = time.perf_counter()

                            pipe.fit(X_train, y_train)

                            t_end = time.perf_counter()
                            print("Modelo treinado")
                            t_train = t_end - t_start

                            predictions = []
                            y_all = []
                            mcc_acc = []
                            
                            for sess in test_sessions:
                                print(f"Testando na sessão {sess}...")

                                raws = self.raw_concat(raws_test[sess])

                                events, event_ids = mne.events_from_annotations(raws)

                                wgen_test = PseudoOnlineWindow(raw=raws,
                                                                    events=events,
                                                                    interval=self.dataset.interval,
                                                                    task_ids=event_ids,
                                                                    window_size=self.wsize,
                                                                    window_step=self.wstep
                                                                    )
                                    
                                X_test, y_test, times_test = wgen_test.generate_windows()

                                predictions_sess = []

                                print("X_train:", len(X_train), "X_test:", len(X_test))
                                print("y_train:", len(y_train), "y_test:", len(y_test))


                                for window in range(len(X_test)):
                                    print(f"testando na janela {window}")
                                    #tempos de inicio e fim da janela, pode ser util pra plot
                                    window_start = times_test[window][0]
                                    window_end = times_test[window][1]


                                    t_start = time.perf_counter()
                                    y_pred = pipe.predict([X_test[window]])[0]
                                    t_end = time.perf_counter()

                                    t_predict = t_end - t_start

                                    print(f"tempor de predict: {t_predict}")

                                    predictions_sess.append(y_pred)
                                    predictions.append(y_pred)
                                    y_all.append(y_test[window])

                                    print("calculando score")
                                    t_start1 = time.perf_counter()
                                    mcc_acc_sess = matthews_corrcoef(y_test[:window+1], predictions_sess[:window+1])    #score acumulado dentro da sessão
                                    mcc_acc = matthews_corrcoef(y_all, predictions)   #score acumulado entre sessões
                                    t_end1 = time.perf_counter()

                                    t_predict1 = t_end1 - t_start1
                                    print(f"tempo de calculo de score: {t_predict1}")


                                    res = {
                                    "dataset": self.dataset,
                                    "subject": subject,
                                    "session": sess,
                                    "method": self.method,
                                    "pipeline": name,
                                    "t_train": t_train,
                                    "window": window,
                                    "window_start": window_start,
                                    "window_end": window_end,
                                    "t_predict": t_predict,
                                    "y_pred": y_pred,
                                    "y_true": y_test[window],
                                    "correct": (predictions_sess[window] == y_test[window]),
                                    "mcc_acc_sess": mcc_acc_sess,
                                    "mcc_acc": mcc_acc
                                    }

                                    self.results_.append(res)
                    else:
                        raise ValueError("Não há sessões suficientes para inter-session")
        if len(self.results_):
            self.results_ = pd.DataFrame(self.results_)
            self.results_.to_csv("pseudo-online-results.csv", index=False)