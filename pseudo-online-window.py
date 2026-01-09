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
    Segments data in windows for pseudo-online analysis.


    Parameters:
        raw: mne.Raw object
            The continuous data.
        events: arra
            MNE event array.
        interval: list
            Dataset parameter defining imagery interval.
        task_ids: dict
            Defines the tasks and its numeric IDs. It can be used to select a subset of the dataset tasks.
        window_size: float
            The window size in seconds.
        window_step: int
            Distance in seconds between the start of two consecutive windows. It can be used to set superposition between windows, when value is lower than window_size.
    
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
        Attributes aa label for each sample. The label vector is initialized with 0 and each data point is attributed to the task label, if it is in imagery period.

        Returns:
            labels: nd array
                Label vector containing labels for each data sample.
        """
        
        n_samples = self.raw.n_times
        labels = np.zeros(n_samples, dtype=int)

        valid_ids = list(self.task_ids.values())

        for ev in self.events:
            ev_idx, _, ev_id = ev

            if ev_id in valid_ids:
                # uses only imagery period for task attribution
                start = ev_idx + self.t_start
                stop = ev_idx + self.t_end

                # ensure array limits
                start = max(0, start)
                stop = min(n_samples, stop)

                labels[start:stop] = ev_id
        return labels


    def generate_windows(self):
        """
        Generates and labels windows.
        
        Returns:
            X: nd array shape=(n_windows, n_channels, n_times)
                The windows (data).
            y: nd array
                Window labels in the same order as X.
            times: nd array
                Array of tuples. Each tuple is the timestamps (start and end) of each window. Might be useful for plotting.
        """
        X, y, times = [], [], []

        data = self.raw.get_data()
        n_samples = data.shape[1]

        for start_idx in range (0, n_samples - self.window_size, self.window_step):
            end_idx = start_idx + self.window_size

            if self.chan_list == None:
                window_data = data[:, start_idx : end_idx]
                window_labels = self.labels[start_idx:end_idx]
            else:   #channel selection
                window_data = []
                for chan in self.chan_list:
                    if chan in self.raw.ch_names:
                        window_data.append(data[chan, start_idx : end_idx])
                    else:
                        raise ValueError(f"Channel {chan} is not in {self.raw.ch_names}")
                window_labels = self.labels[start_idx:end_idx]

            count = np.bincount(window_labels)
            major = np.argmax(count)

            prop_major = count[major] / len(window_labels)

            # class draw proportion
            n_classes = len(np.unique(window_labels))
            draw_prop = 1 / n_classes

            # in case of draw, the posterior label wins
            if prop_major != draw_prop:
                y.append(major)
            else:
                y.append(window_labels[-1])

            X.append(window_data)
            times.append(((start_idx / self.sfreq), (end_idx / self.sfreq)))

        return np.array(X), np.array(y), np.array(times)


class IdleBaseline(BaseEstimator, TransformerMixin):
    """
    Applies baseline correction. It uses a fixed baseline. This transformer has to be applied to windowed data.

    Parameters:
        rest_label: int
            Label representing idle state.
    
    Returns:
        X: nd array shape=(n_windows, n_channels, n_times)
            Baseline corrected windowed data.
    """

    def __init__(self, rest_label=0):
        self.rest_label = 0

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Missing labels array")
        
        idle_windows = (y == self.rest_label)

        if not np.any(idle_windows):
            raise ValueError("There are no rest windows")
        
        self.baseline_ = X[idle_windows].mean(axis=0)
        return self

    def transform(self, X):
        return np.subtract(X, self.baseline_)


class PSD(BaseEstimator, TransformerMixin):

    def __init__(self, sfreq, fmin=0, fmax=None):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax

    def fit (self, X, y=None):
        return self

    def transform(self, X):
        
        fmax = self.fmax if self.fmax is not None else self.sfreq / 2

        psds, _ = mne.time_frequency.psd_array_welch(X, 
                                                     sfreq=self.sfreq,
                                                     fmin=self.fmin,
                                                     fmax=self.fmax,
                                                     average='mean')
        
        return psds
        
class IdleDetection():
    """
    Classifier for idle state detection.
    """
    def __init__():
        pass

    def fit(self, X, y=None):
        pass

    def predict(self, X, y):
        pass


class PseudoOnlineEvaluation():
    """
    Evaluates windowed data in pseudo-online manner (simulating real time asynchronous BCI). It can be done within-session, evaluating only one session data, or inter-session, evaluating performance across sessions, without
    breaking signal causality.

    Parameters:
        dataset: MOABB dataset
            Dataset to be used. It has to be a MOABB dataset object.
        pipelines: dict
            Sklearn pipelines dictionary, as used in MOABB.
        method: string
            Either 'within-session' or 'inter-session'.
                within-session: trains models on first k windows, testing on the remaining ones.
                inter-session: trains models on first k sessions, testing on the remaining ones; doesn't violate data causality.
        ratio: float
            Proportion of data to be used in training.
        
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
        Auxiliary function for raw data concat.
        """
        if len(raw_list) == 0:
            raise ValueError("Raw list is empty.")
        elif len(raw_list) == 1:    #aqui, o raw de uma sessao pode ser constituido de 1 ou mais runs, por isso essa verificação
            if type(raw_list[0]) != list:
                return raw_list[0]
            else:
                return mne.concatenate_raws(raw_list[0])
        else:
            return mne.concatenate_raws(raw_list)

    def evaluate(self):
        """
        Main function for processing data.
        """

        for subject in self.subjects:
            if subject not in self.dataset.subject_list:
                raise ValueError(f"Invalid subject index: {subject}")
            else:
                print(f"Processing subject {subject}...")

                raws_dict = {}
                raws_test = {}
                pre = self.dataset.get_data(subjects=[subject])
                
                session_keys = []   # stores session ids (not always int)

                if self.method == 'within-session':
                    
                    # raw extraction from moabb dataset
                    for _, runs in pre.items():
                        for sess, dicts in runs.items():
                            session_keys.append(sess)
                            raws_dict[sess] = []
                            for _, data in dicts.items():
                                raws_dict[sess].append(data)
                    
                    for sess in session_keys:
                        print(f"Processing session {sess} subject {subject}...")
                        raw = self.raw_concat(raws_dict[sess])
                        events, event_ids = mne.events_from_annotations(raw)

                        wgen = PseudoOnlineWindow(raw=raw,
                                                events=events,
                                                interval=self.dataset.interval,
                                                task_ids=event_ids,
                                                window_size=self.wsize,
                                                window_step=self.wstep
                                                )
                        
                        X, y, times = wgen.generate_windows()

                        idx_split = int(len(X) * self.ratio)

                        times_test = times[idx_split:]

                        X_train, y_train = X[:idx_split], y[:idx_split]
                        X_test, y_test = X[idx_split:], y[idx_split:]

                        if (self.no_run):
                            return X_train, y_train, X_test, y_test

                        for name, pipe in self.pipelines.items():
                            t_start = time.perf_counter()

                            print("Fitting...")
                            pipe.fit(X_train, y_train)
                            print("Done fitting!")

                            t_end = time.perf_counter()
                            t_train = t_end - t_start


                            print(f"Fitting time: {t_train}")

                            predictions = []
                            mcc_acc = []

                            for window in range(len(X_test)):
                                window_start = times_test[window][0]
                                window_end = times_test[window][1]

                                t_start = time.perf_counter()

                                y_pred = pipe.predict([X_test[window]])[0]

                                t_end = time.perf_counter()

                                predictions.append(y_pred)
                                t_predict = t_end - t_start

                                mcc_acc = matthews_corrcoef(y_test[:window+1], predictions[:window+1]) # accumulated mcc score until this window

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
                        session_split = int(self.ratio * self.dataset.n_sessions)
                        raws_list = []
                        raws_train = []

                        print(f"Splitting index is {session_split}, dataset has {self.dataset.n_sessions} sessions per subject.")

                        for _, runs in pre.items():
                            for sess, dicts in runs.items():
                                session_keys.append(sess)
                                raws_test[sess] = []
                                raws_dict[sess] = []
                                for _, data in dicts.items():
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
                            
                        print(f"Fitting in sessions: {train_sessions}...")

                        for name, pipe in self.pipelines.items():
                            print(f"Pipeline: {name}")
                            print("FItting...")
                            t_start = time.perf_counter()

                            pipe.fit(X_train, y_train)

                            t_end = time.perf_counter()
                            print("Done fitting!")
                            t_train = t_end - t_start

                            predictions = []
                            y_all = []
                            mcc_acc = []
                            
                            for sess in test_sessions:
                                print(f"Testing in session {sess}...")

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
                                    print(f"Testing on window {window}")
                                    #tempos de inicio e fim da janela, pode ser util pra plot
                                    window_start = times_test[window][0]
                                    window_end = times_test[window][1]


                                    t_start = time.perf_counter()
                                    y_pred = pipe.predict([X_test[window]])[0]
                                    t_end = time.perf_counter()

                                    t_predict = t_end - t_start

                                    predictions_sess.append(y_pred)
                                    predictions.append(y_pred)
                                    y_all.append(y_test[window])

                                    mcc_acc_sess = matthews_corrcoef(y_test[:window+1], predictions_sess[:window+1])    #score acumulado dentro da sessão
                                    mcc_acc = matthews_corrcoef(y_all, predictions)   #score acumulado entre sessões

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
                        raise ValueError("There are not enough sessions for evaluation.")
        if len(self.results_):
            self.results_ = pd.DataFrame(self.results_)
            self.results_.to_csv(f"results-S{subject}.csv", index=False)