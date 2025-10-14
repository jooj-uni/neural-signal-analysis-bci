#%%
import    mne
import    scipy.io          as sio
import    numpy             as np
import    pandas            as pd
import    matplotlib.pyplot as plt
from   mne                           import Epochs, pick_types
from   mne_icalabel                  import label_components
from   mne.preprocessing             import ICA
from   mne.decoding                  import CSP
from   sklearn.pipeline              import Pipeline
from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from   sklearn.model_selection       import ShuffleSplit, cross_val_score, train_test_split
from   autoreject                    import compute_thresholds

#%% EPOCAGEM E FILTRAGEM INICIAL DOS DADOS

sujeitos = [1, 2]
dados = {}
raws = {}
noise_chan = {}

for suj in sujeitos:
    path_data         = 'C:/Users/aihon/Desktop/proc_EEG/'
    sess              = 2
    filename          = f'S{suj}_Session_{sess}.mat'
    ind_names         = ['data', 'time', 'positionx', 'positiony', 'SRATE', 'TrialData', 'metadata', 'chaninfo']
    trial_data_labels = ['tasknumber', 'runnumber', 'trialnumber', 'targetnumber', 'triallength',
                         'targethitnumber', 'resultind', 'result', 'forcedresult', 'artifact']
    feedback_len_s    = 3.5
    trial_len_s       = 7.5
    
    data              = sio.loadmat(path_data + filename)
    BCI_data          = data['BCI'][0][0]
    sampling_freq     = BCI_data[ind_names.index('SRATE')][0][0]
    raw_trials        = BCI_data[ind_names.index('TrialData')][0]
    channels_info     = BCI_data[ind_names.index('chaninfo')][0][0]
    noise_chan[suj]   = BCI_data[ind_names.index('chaninfo')][0][0][1]
    
    # Nomes e posições dos canais
    ch_names          = [str(c[0]) for c in channels_info[0][0]]
    positions         = np.array([[c[1][0][0], c[2][0][0], c[3][0][0]] for c in channels_info[2][0]])
    montage           = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, positions)), coord_frame='head')
    info              = mne.create_info(ch_names=ch_names, sfreq=sampling_freq, ch_types='eeg')
    info.set_montage(montage)
    
    # Processamento dos dados dos trials
    df                = pd.DataFrame(raw_trials, columns=trial_data_labels).dropna()
    for col in trial_data_labels:
      df[col]         = df[col].apply(lambda x: x[0][0])
    df_clean          = df[(df['triallength'] >= feedback_len_s) & (~df['targetnumber'].isin([3, 4]))]
    
    # Construção do sinal contínuo
    trial_samples     = int(trial_len_s * sampling_freq)
    all_data          = [BCI_data[ind_names.index('data')][0][int(r['trialnumber']) - 1][:, :trial_samples] for _, r in df_clean.iterrows()]
    epochs_data       = np.concatenate(all_data, axis=1)
    epochs_data = epochs_data * 1e-6
    
    # Criação do RawArray e pré-processamento
    raw               = mne.io.RawArray(epochs_data, info)
    raw.set_eeg_reference(ref_channels='average')
    
    raws[suj] = raw
    
    # EPOCAGEM
    # Eventos e criação dos Epochs
    target_labels    = df_clean['targetnumber'].astype(int).to_numpy()
    
    cue_idx = 2001
    trial_starts = np.arange(0, len(raw.times), trial_samples)
    events_idx = trial_starts + cue_idx

    events           = np.column_stack((events_idx, np.zeros(len(target_labels), dtype=int), target_labels))
    event_id         = dict(right=1, left=2)
    picks            = pick_types(raw.info, eeg=True)
    epochs           = Epochs(raw, events.astype(int), event_id=event_id, picks=picks, preload=True, baseline=(0, 0), tmin=-4, tmax=3.5)
    
    epochs.filter(0.5, 50 , fir_design='firwin', skip_by_annotation='edge')
    
    dados[suj] = epochs
    
    epochs.plot()

#%% PRE-PROCESSAMENTO
#==== REJEIÇÃO POR LIMIAR COM AUTOREJECT SEM INTERPOLAÇÃO ====
thresholds = {}
bad_epochs = {}

for suj in sujeitos:
    data = dados[suj].get_data()
    thresholds[suj] = compute_thresholds(dados[suj], method='random_search', random_state=42, verbose=True)

    bad_epochs[suj] = []

    for i, trial in enumerate(data):
        for ch_idx, ch_name in enumerate(ch_names):
            if np.max(np.abs(trial[ch_idx])) > thresholds[suj][ch_name]:
                bad_epochs[suj].append(i)
                break
    dados[suj] = dados[suj].drop(bad_epochs[suj])
    
    print(f"Trials rejeitadas sujeito {suj}: {len(bad_epochs[suj])}")
    
    dados[suj].plot()

#%% ANÁLISES INTRASUJEITO
#==== CSP+LDA: VALIDAÇÃO CRUZADA E ANÁLISE DE JANELA TEMPORAL ====

chance_level = {}
scores_cv={}

for suj in sujeitos:
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    x = dados[suj].get_data()
    y = dados[suj].events[:, -1]
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scores_cv[suj] = cross_val_score(clf, x, y, cv=cv)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    csp.fit(x_train, y_train)
    lda.fit(csp.transform(x_train), y_train)
    
    sfreq            = sampling_freq
    w_length         = int(sfreq * 0.5)
    w_step           = int(sfreq * 0.1)
    w_start          = np.arange(0, x_test.shape[2] - w_length, w_step)
    
    score_this_window = []
    for n in w_start:
      X_csp_win      = csp.transform(x_test[:, :, n:(n + w_length)])
      acc_win        = lda.score(X_csp_win, y_test)
      score_this_window.append(acc_win)
    w_times          = (w_start + w_length / 2.) / sfreq + dados[suj].tmin
    
    classes, contagem = np.unique(y, return_counts=True)
    idx_max = np.argmax(contagem)
    cont_max = contagem[idx_max]
    class_max = classes[idx_max]
    total_eventos = len(y)
    chance_level[suj] = cont_max/total_eventos
        
    print(f"Tentativas Mão Direita: {np.sum((y==1))}")
    print(f"Tentativas Mão Esquerda: {np.sum((y==2))}")
    print(f"Acurácia média (CV sessão {sess}): {np.mean(scores_cv[suj]):.3f} ± {np.std(scores_cv[suj]):.3f}")
    print(f"Acurácia final (janela completa): {lda.score(csp.transform(x_test), y_test):.3f}")
    
    plt.figure(figsize=(15, 5))
    plt.plot(w_times, score_this_window, label='Score')
    plt.axvline(0, linestyle='-', color='k', label='Onset')
    plt.axhline(0.5, linestyle='--', color='k', label='Chance')
    plt.axhline(chance_level[suj], linestyle='--', color='k', label='Chance level')
    plt.xlabel('Time (s)')
    plt.ylabel('Classification Accuracy')
    plt.title(f"Classification accuracy over time subject {suj} session {sess}")
    plt.legend(loc='lower right')
    plt.ylim(0,1)
    plt.show()
        
    
    
    
    
    
    