import numpy as np
import moabb
import mne
from sklearn.metrics import matthew_corrcoef


class PseudoOnlineWindow():
    """
    cria janelas deslizantes e as rotula, baseado no arigo do framework pseudo online
    os rotulos sao dados de acordo com a classe majoritaria da janela
    ==============================
    raw: objeto mne.Raw
    events: array de eventos padrao do mne
    interval: parametro do dataset que define os intervalos de imagetica
    task_ids: define quais os ids das tasks que vao ser classificadas (permite problema multiclasse ou one-versus-rest, por exemplo)
    window_size: define tamanho (em segundos) da janela
    window_step: distancia entre os inicios de duas janelas adjacentes, entao define a sobreposicao entre janelas

    *******************************arrumar para logica funcionar por amostras e nao tempo************************************
    """
    def __init__(self, raw, events, interval, task_ids, window_size, window_step):
        self.raw = raw
        self.events = events
        self.interval = interval
        self.sfreq = raw.info['sfreq']
        self.task_ids = task_ids

        self.window_size = int(window_size * self.sfreq)
        self.window_step = int(window_step * self.sfreq)
        self.t_start = int(interval[0] * self.sfreq)
        self.t_end = int(interval[1] * self.sfreq)

        self.labels = self.generate_labels

    def generate_labels(self, start, stop):
        """
        atribui uma classe para cada amostra do dado. inicializa o vetor de rotulos em 0 e atribui a classe da task
        às amostras do período de imagética
        """
        
        #aqui n_sample eh de uma forma e la embaixo de outra
        n_samples = self.raw.n_times
        labels = np.zero(n_samples, dtype=int)

        valid_ids = list(self.event_id.values()) #vai selecionar so os eventos que queremos

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

        data = self.raw.get_data()
        #n_samples ta sendo obtido de outra forma la em cima
        n_samples = data.shape[1]

        for start_idx in range (0, n_samples - self.window_size, self.window_step):
            end_idx = start_idx + self.window_size

            window_data = data[:, start_idx : end_idx]
            window_labels = self.labels[start_idx:end_idx]

            count = np.bincount(window_labels)
            major = np.argmax(count)

            prop_major = count[major] / len(window_labels)

            if prop_major != 0.5:
                y.append(major)
            #se ha empate, vence a classe posterior
            else:
                y.append(window_labels[-1])

            X.append(window_data)
            times.append(start_idx / self.sfreq)

        return np.array(X), np.array(y), np.array(times)

class PseudoOnlineEvaluation():
    """
    faz avaliacao com janelas deslizantes, tanto na mesma sessao quanto inter sessao.
    =========================================
    dataset: dataset utilizado
    pipelines: dict de pipelines do moabb
    method: pode ser 'within-session' para avaliacao na mesma sessao ou 'inter-session' para avaliacao entre sessoes
        within session: treina nas primeiras k trials definidas por ratio e testa nas demais, dentro de uma unica sessao
        inter session: treina nas prmeiras k sessoes definidas por ratio e testa nas demais sessoes
    ratio: define a proporçao dos dados usada para treino
    
    """
    def __init__(self, dataset, pipelines, method, ratio=0.7):
        self.dataset = dataset
        self.pipelines = pipelines
        self.ratio = 0.7
        self.method = method
        self.y_ = {}
        self.mscores = {}
    
    def evaluate(self):
        """
        algumas questoes:
            a extracao do array de events ainda nao ta clara e a funcao supoe ela. mesma coisa para task_ids
            a extracao de dados de um dataset moabb ainda nao ta tao clara tambem

        ********** checar os loops sobre dataset.get_data().items() e se ta trabalhando corretamente com objetos Raw *************
        ********** atualizar o array de eventos quando append(raw)
        """
        events = []
        task_ids = []
        wsize = []
        wstep = []

        if self.method == 'within-session':
            #checar se subject_id = subject_list
            for subject in self.dataset.subject_list:
                raw = []
                pre = self.dataset.get_data(subject=subject)
                #checar se a extracao de dados ta correta
                for sess, run in pre.items():
                    for run, data in run.items():
                        raw.append(data)
                        #aqui extracao de eventos
                        wgen = PseudoOnlineWindow(raw=raw,
                                                events=events,
                                                task_ids=task_ids,
                                                window_size=wsize,
                                                window_step=wstep)
                        X, y, times = wgen.generate_windows()

                        idx_split = int(len(X) * self.ratio)

                        X_train, y_train = X[:idx_split], y[:idx_split]
                        X_test, y_test = X[idx_split:], y[idx_split:]

                        for name, pipe in self.pipelines.items():
                            pipe.fit(X_train, y_train)
                            self.y_[name] = pipe.predict(X_test, y_test)

                            self.mscores[name] = matthew_corrcoef(y_test, self.y_[name])

        elif self.method == 'inter-session':
            self.y_ = {}
            self.mscores = {}
            #checar se subject_id = subject_list
            for subject in self.dataset.subject_list:
                raw = []
                pre = self.dataset.get_data(subject=subject)
                keys = pre.keys()
                idx_split = int(len(keys * self.ratio))

                #checar se a extracao de dados ta correta
                for sess, run in pre.items():
                    if keys.index(sess) < idx_split:
                        for data in run.items():
                            raw.append(data)
                
                wgen_train = PseudoOnlineWindow(raw=raw,
                                                events=events,
                                                window_size=wsize,
                                                window_step=wstep)
                X_train, y_train, times_train = wgen_train.process()

                for sess, run in pre.items():
                    raw = []
                    if keys.index(sess) >= idx_split:
                        for data in run.items():
                            raw.append(data)
                        
                        wgen_sesstest = PseudoOnlineWindow(raw=raw,
                                                           events=events,
                                                           window_size=wsize,
                                                           window_step=wstep)
                        X_sesstest, y_sesstest, times_sesstest = wgen_sesstest.process()

                        for name, pipe in self.pipelines.items():
                            pipe.fit(X_train, y_train)

                            self.y_[sess][name] = pipe.predict(X_sesstest, y_sesstest)

                            self.mscores[sess][name] = matthew_corrcoef(y_test, self.y_[sess][name])
        return self.mscores
        










                    