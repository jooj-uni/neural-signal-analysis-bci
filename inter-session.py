import time
import logging
from sklearn.base import clone
from sklearn.metrics import get_scorer
from moabb.evaluations.base import BaseEvaluation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from mne.epochs import BaseEpochs
from moabb.evaluations.utils import (
    _create_save_path,
    _save_model_cv,
)


try:
    from codecarbon import EmissionsTracker

    _carbonfootprint = True
except ImportError:
    _carbonfootprint = False

log = logging.getLogger(__name__)

class InterSessionEvaluation(BaseEvaluation):
    """Cross-session performance evaluation.
    A princípio, ele tá treinando nas duas primeiras sessões e testando nas seguintes. 
    """

    # flake8: noqa: C901
    def evaluate(
        self, dataset, pipelines, param_grid, process_pipeline, postprocess_pipeline=None
    ):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
            # Progressbar at subject level
        for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CrossSession"):
            # check if we already have result for this subject/pipeline
            # we might need a better granularity, if we query the DB
            run_pipes = self.results.not_yet_computed(
                pipelines, dataset, subject, process_pipeline
            )
            if len(run_pipes) == 0:
                log.info(f"Subject {subject} already processed")
                continue

            # get the data
            X, y, metadata = self.paradigm.get_data(
                dataset=dataset,
                subjects=[subject],
                return_epochs=self.return_epochs,
                return_raws=self.return_raws,
                cache_config=self.cache_config,
                postprocess_pipeline=postprocess_pipeline,
                process_pipelines=[process_pipeline],
            )


            
            le = LabelEncoder()
            y = y if self.mne_labels else le.fit_transform(y)
            groups = metadata.session.values
            scorer = get_scorer(self.paradigm.scoring)

            for name, clf in run_pipes.items():
                if _carbonfootprint:
                    # Initialise CodeCarbon
                    tracker = EmissionsTracker(save_to_file=False, log_level="error")
                    tracker.start()

                inner_cv = StratifiedKFold(
                    3, shuffle=True, random_state=self.random_state
                )

                grid_clf = clone(clf)

                # Implement Grid Search
                grid_clf = self._grid_search(
                    param_grid=param_grid, name=name, grid_clf=grid_clf, inner_cv=inner_cv
                )

                if self.hdf5_path is not None and self.save_model:
                    model_save_path = _create_save_path(
                        hdf5_path=self.hdf5_path,
                        code=dataset.code,
                        subject=subject,
                        session="",
                        name=name,
                        grid=self.search,
                        eval_type="CrossSession",
                    )

                sessions = sorted(metadata.session.unique())
                n_train_sessions = 2
                train_sessions = sessions[:n_train_sessions]
                test_sessions = sessions[n_train_sessions:]

                train_idx = metadata.index[metadata.session.isin(train_sessions)].to_numpy()
                test_idx = metadata.index[metadata.session.isin(test_sessions)].to_numpy()

                for cv_ind, (train, test) in enumerate([(train_idx, test_idx)]):
                    model_list = []
                    if _carbonfootprint:
                        tracker.start()
                    t_start = time()

                    cvclf = clone(grid_clf)

                    cvclf.fit(X[train], y[train])

                    model_list.append(cvclf)
                    score = scorer(cvclf, X[test], y[test])

                    if self.hdf5_path is not None and self.save_model:
                        _save_model_cv(
                            model=cvclf,
                            save_path=model_save_path,
                            cv_index=str(cv_ind),
                        )

                    if _carbonfootprint:
                        emissions = tracker.stop()
                        if emissions is None:
                            emissions = 0

                    duration = time() - t_start

                    nchan = X.info["nchan"] if isinstance(X, BaseEpochs) else X.shape[1]
                    res = {
                        "time": duration,
                        "dataset": dataset,
                        "subject": subject,
                        "train_sessions": train_sessions,
                        "test_sessions": test_sessions,
                        "score": score,
                        "n_samples": len(train),
                        "n_channels": nchan,
                        "pipeline": name,
                    }
                    if _carbonfootprint:
                        res["carbon_emission"] = (1000 * emissions,)

                    yield res

    def is_valid(self, dataset):
        return dataset.n_sessions > 1