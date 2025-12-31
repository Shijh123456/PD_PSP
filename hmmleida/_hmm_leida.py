import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import Bunch
from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

from .dynamics_metrics import (
    group_transition_matrix,
    compute_dynamics_metrics,
)
from .data_utils import (
    load_tseries,
    load_classes,
    load_rois_labels,
    load_rois_coordinates,
    datapath_processing
)
from .clustering import (
    hmm_analysis,
    identify_states,
    plot_clusters3D,
    centroid2matrix,
    plot_voronoi,
    barplot_states,
    barplot_eig,
    plot_clustering_scores
)
from .clustering import rsnets_overlap as rsnets
from .plotting import (
    brain_states_nodes,
    brain_states_network,
    states_in_bold,
    plot_pyramid,
    states_k_glass,
    brain_states_on_surf,
    brain_states_on_surf2,
    _explore_state,
    _save_html
)
from .signal_tools import (
    hilbert_phase,
    phase_coherence,
    get_eigenvectors
)
from .data_utils.validation import (
    _check_k_input,
    _check_metric,
    _check_state
)
from .stats import _compute_stats,scatter_pvalues

class hmmLeida:
    def __init__(self,atlas_space, output_path):
        if not isinstance(atlas_space,str):
            raise TypeError("'data_path' must be a string!")

        group  = ['HC', 'PD', 'PSP']
        #group = ['HC', 'PD']
        data_path = datapath_processing(atlas_space, group, output_path)

        self.time_series = load_tseries(data_path)
        self.classes = load_classes(data_path)
        self.rois_labels = load_rois_labels(data_path)
        self.rois_coordinates = load_rois_coordinates(data_path)
        self.atlas = atlas_space
        self._results_path_ = f'{output_path}/result/hmm_result/results_{self.atlas}_age'

        #self._validate_constructor_params()

    def fit_predict(self, TR=None, paired_tests=False, n_perm=5_000, save_results=True, random_state=None):
        self._K_min_ = 2
        self._K_max_ = 20

        self.concatenated_data, self._clustering_, self._dynamics_ = self._execute_all(
            TR=TR,
            random_state=random_state,
            paired_tests=paired_tests,
            n_perm=n_perm,
            save_results=save_results,
        )

        self.predictions = self._clustering_.predictions
        self._classes_lst_ = np.unique(self.concatenated_data.condition).tolist()
        self._N_classes_ = len(self._classes_lst_)
        self._is_fitted = True


    def _execute_all(self, TR=None, random_state=None, paired_tests=False, n_perm=5_000, save_results=True):
        subject_ids = list(self.time_series.keys())
        N_subjects = len(subject_ids)

        if save_results:
            if os.path.exists(self._results_path_):
                raise Warning(f"EXECUTION ABORTED: The folder {self._results_path_} already "
                            "exists. If you have results from earlier executions of "
                            "the analysis, consider changing the folder's name or moving "
                            "the folder to another location.")
            else:
                try:
                    print(f"\n-Creating folder to save results: './{self._results_path_}'")
                    os.makedirs(self._results_path_)
                except:
                    raise Exception("The folder to save the results could't be created.")


        sub_list, class_list, concatenated_data = [], [], []

        print("\n-STARTING THE PROCESS:\n"
             "========================\n"
             f"-Number of subjects: {N_subjects}")

        print("\n 1) concatenate the BOLD data across the multiple subjects.\n")
        for sub_idx, sub_id in enumerate(subject_ids):
            tseries = self.time_series[sub_id]
            N_volumes = tseries.shape[1]
            print(f"SUBJECT ID: {sub_id} ({tseries.shape[1]} volumes)")
            newtseries = tseries[1:,:]
            if len(concatenated_data) == 0:
                concatenated_data = np.row_stack(
                    newtseries.T)
            else:
                concatenated_data = np.row_stack((concatenated_data,
                                       newtseries.T))

            for volume in range(N_volumes):
                sub_list.append(sub_id)
                if len(self.classes[sub_id]) > 1:
                    class_list.append(self.classes[sub_id][volume + 1])
                else:
                    class_list.append(self.classes[sub_id][0])

        concatenated_dataset = pd.DataFrame(np.vstack(concatenated_data), columns=self.rois_labels)
        concatenated_dataset.insert(0, 'subject_id', sub_list)
        concatenated_dataset.insert(1, 'condition', class_list)

        if save_results:
            try:
                concatenated_dataset.to_csv(f'{self._results_path_}/concatenated_data.csv',sep='\t',index=False)
            except:
                print("Warning: An error ocurred when saving the 'concatenated_data.csv' file to local folder.")

        print("\n 2) use hmmlearn analysis BOLD signals.\n")
        predictions, clustering_performance, models = hmm_analysis(
            concatenated_dataset,
            K_min=self._K_min_,
            K_max=self._K_max_,
            random_state=random_state,
            save_results=save_results,
            path=self._results_path_ if save_results else None
        )

        # computing dynamical systems theory metrics for each K
        print("\n 3) COMPUTING THE DYNAMICAL SYSTEMS THEORY METRICS FOR EACH K.")
        dynamics_data = compute_dynamics_metrics(
            predictions,
            TR=TR,
            save_results=save_results,
            path=self._results_path_ if save_results else None
        )

        # Statistical analysis of occupancies and dwell times for each k
        print("\n 4) EXECUTING THE STATISTICAL ANALYSIS OF "
              "OCCUPANCIES AND DWELL TIMES FOR EACH K.")
        stats = _compute_stats(
            dynamics_data,
            paired_tests=paired_tests,
            n_perm=n_perm,
            save_results=save_results,
            path=self._results_path_ if save_results else None
        )

        print("\n-Creating figures with the statistical analyses results for dwell times "
              "and fractional occupancies. This may take some time. Please wait...")

        classes = np.unique(concatenated_dataset.condition)

        for metric in ['occupancies', 'dwell_times']:
            pooled_stats = pd.concat((stats[metric]), ignore_index=True)

            for conditions in combinations(classes, 2):
                pooled_stats_ = pooled_stats[
                    (pooled_stats.group_1.isin(conditions))
                    &
                    (pooled_stats.group_2.isin(conditions))
                    ].reset_index(drop=True)

                # plot pyramid
                dyn_data = {k: v[v.condition.isin(conditions)] for k, v in dynamics_data[metric].items()}
                plot_pyramid(
                    dyn_data,
                    pooled_stats_,
                    K_min=self._K_min_,
                    K_max=self._K_max_,
                    metric_name=metric,
                    despine=True
                )
                if save_results:
                    plt.savefig(f'{self._results_path_}/dynamics_metrics/{conditions[0]}_vs_{conditions[-1]}_{metric}_barplot_pyramid.png',dpi=300)

                # plot p-values scatter plot
                scatter_pvalues(pooled_stats_, metric=metric, fill_areas=True)
                if save_results:
                    plt.savefig(f'{self._results_path_}/dynamics_metrics/{conditions[0]}_vs_{conditions[-1]}_{metric}_scatter_pvalues.png',dpi=300)

        # Preparing output
        clustering = Bunch(
                    predictions=predictions,
                    performance=clustering_performance,
                    models=models
                )

        dynamics = Bunch(
                    dwell_times=dynamics_data['dwell_times'],
                    occupancies=dynamics_data['occupancies'],
                    transitions=dynamics_data['transitions'],
                    stats=stats
                )

        print("\n** THE ANALYSIS HAS FINISHED SUCCESFULLY!")
        if save_results:
            print(f"-All the results were save in './{self._results_path_}'")

        print("\n-You can explore the results in detail by using "
                      "the methods and attributes of the Leida class.")

        return concatenated_dataset, clustering, dynamics




