import pandas as pd
import numpy as np
import pickle
import os
import shutil

#General functions

def datapath_processing(atlas_space, group, output_path):
    input_path = '/zebra/Beijing_Xuanwu/Wutao_PD_RBD/PD_PSP_age'
    #input_path = 'D:/shuoshi/PD/data'
    #data_path = f'D:/shuoshi/PD/data/input_{atlas_space}'
    data_path = f'/data/sjh/PD_PSP_input_{atlas_space}_age'
    i = 1
    sub_list, condition_list, sub_number = [], [], []
    for item in group:
        signals_path = f'{input_path}/{item}/fmripost'
        sub_folders = [f for f in os.listdir(signals_path) if os.path.isdir(f'{signals_path}/{f}')]
        sub_folders.sort()
        for sub in sub_folders:
            if sub == 'sub-061' and item == 'HC':
                continue
            for file in os.listdir(f'{signals_path}/{sub}'):
                if item == 'PSP':
                    if file.endswith('.csv') and 'BOLD' in file and f'{atlas_space}x7_MNI' in file and 'run-02' not in file:
                        src_file = f'{signals_path}/{sub}/{file}'
                        dst_folder = f'{data_path}/time_series/sub-{str(i).zfill(3)}'
                        os.makedirs(dst_folder, exist_ok=True)
                        dst_file = os.path.join(dst_folder, f'sub-{str(i).zfill(3)}.csv')
                        df = pd.read_csv(src_file)
                        df = df.iloc[:, 1:]
                        df = df.T
                        df.to_csv(dst_file, index=False)
                        sub_list.append(f'sub-{str(i).zfill(3)}')
                        condition_list.append(item)
                if file.endswith('.csv')and 'BOLD'in file and f'{atlas_space}_MNI' in file and 'run-02' not in file:
                    src_file = f'{signals_path}/{sub}/{file}'
                    dst_folder = f'{data_path}/time_series/sub-{str(i).zfill(3)}'
                    os.makedirs(dst_folder, exist_ok=True)
                    dst_file = os.path.join(dst_folder, f'sub-{str(i).zfill(3)}.csv')
                    df = pd.read_csv(src_file)
                    df = df.iloc[:, 1:]
                    df = df.T
                    df.to_csv(dst_file, index=False)
                    sub_list.append(f'sub-{str(i).zfill(3)}')
                    condition_list.append(item)

            i = i + 1
            sub_number.append(sub)
    metadata = pd.DataFrame({
        'subject_id': sub_list,
        'condition': condition_list,
        'sub_number': sub_number
    })
    metadata.to_csv(f'{data_path}/metadata.csv', index=False, header=True)
    for file in os.listdir(f'{output_path}/atlas/{atlas_space}'):
        try:
            shutil.copy2(f'{output_path}/atlas/{atlas_space}/{file}', data_path)
        except:
            print("the labels cannot load")
    return data_path

def array2dict(array,subject_ids):
    """
    Convert a 3D array with shape 
    (N_ROIs,N_volumes,N_subjects)
    to a dictionary representation.

    Params:
    -------
    array : ndarray with shape (N_ROIs,N_volumes,N_subjects). 
        Contains the BOLD time series of a group of
        subjects.

    subject_ids : list or array. 
        Contain the subjects' ids (in the same order
        that they appear in the array with the signals).
    
    Returns:
    --------
    dict_data : dict. 
        Contains the subjects ids as keys, and the
        BOLD time series as values.
    """
    dict_data = {}
    for subject_idx,subject in enumerate(subject_ids):
        dict_data[subject] = array[:,:,subject_idx]
    return dict_data

def load_dictionary(filepath):
    """
    Load dictionary from pickle (.pkl)
    file in local folder.
    
    Params:
    --------
    filepath : str.
        Specify the path to the pickle
        file to be loaded.

    Returns:
    --------
    dict_ : dict.
        Loaded dictionary.
    """
    with open(filepath, 'rb') as file:
        dict_ = pickle.load(file)
        return dict_

def save_dictionary(filename,dictionary):
    """
    Save dictionary in local folder
    as a pickle (.pkl) file.
    
    Params:
    --------
    filename : str.
        Specify the name (and optionally the
        path) of the pickle file to be saved.
    """
    with open(f'{filename}.pkl', 'wb') as file:
        pickle.dump(dictionary, file)

def list2txt(list_,filename=None):
    """
    Save a list as a .txt file.
    Elements are located line by line.

    Params:
    -------
    list_ : list.
        The list to be saved on local folder.

    filename : str.
        Specify the name of the file to be saved.
    """
    filename = 'txt_from_list.txt' if filename is None else f'{filename}.txt'
    with open(f'{filename}', 'w') as f:
        for line in list_:
            f.write(line)
            f.write('\n')

def txt2list(path):
    """
    Load a .txt file as a list.
    Note: the .txt file must contain
    an entry/value per line/row.

    Params:
    -------
    path : str.
        Full path to the .txt file of
        interest. E.g.: 'data/rois_labels.txt'

    Returns:
    --------
    list_ : list.
    """
    with open(path,'r') as file:
        list_ = [line.strip() for line in file]
    return list_

#load output data from local
def load_model(k=2,models_path=None):
    """
    Load model from .pkl file.
    
    Params:
    -------
    k : int.
        Select the model to load.

    models_path : str.
        Path the folder that contains the
        saved models for each k partition.
    
    Returns:
    --------
    model : KMeansLeida object.
        The fitted model that was used to
        predict the cluster labels of each
        observation
    """
    if models_path is None:
        print("You must provide a path to the models folder.")
    else:
        try:
            model = pd.read_pickle(f'{models_path}/model_k_{k}.pkl')
            return model
        except:
            raise Warning("The model couldn't be loaded. Check that the '.pkl' "
                        "file is located in the provided 'path'.")

#Load input data
def load_classes(path):
    """
    Load the .csv or .pkl 'metadata' file that contains
    the label/s specifying the group/condition to which
    each subject or volume belongs to and return it as a
    dictionary.

    Params:
    -------
    path : str.
        Path to local folder where the
        'metadata' file is located.
    
    Returns:
    --------
    classes : dict
        Dictionary with 'subject_ids' as keys
        and the labels as values.
    """ 
    try:
        try:
            classes = load_dictionary(f'{path}/metadata.pkl')
        except:
            metadata = pd.read_csv(f'{path}/metadata.csv',sep=',')
            cols = ['subject_id','condition']
            if not all(item in list(metadata.columns) for item in cols):
                raise Exception(f'{cols} columns must be present in {path}/metadata.csv')
            classes = {}
            for sub in np.unique(metadata.subject_id):
                classes[sub] = [label for label in metadata[metadata.subject_id == sub].condition]

        return classes
    except:
        raise Exception("The groups/conditions labels coudn't be loaded.")

def load_rois_labels(path):
    """
    Load the .txt file that contains
    the labels of each ROI and return
    it as a list.

    Params:
    -------
    path : str
        Path to the local folder
        in which the file is located.

    Returns:
    --------
    labels : list
        Returns a list with the ROIs labels.
    """ 
    try:
        return txt2list(f'{path}/rois_labels.txt')
    except: 
        raise Exception("The ROIs' labels couldn't be loaded "
                        "from the provided path.")

def load_tseries(path):
    """
    Load the time series of each subject that are contained
    either in a .pkl file in 'path', or in individual
    .csv files in '{path}/time_series/{subject_id}/'.

    Params:
    -------
    path : str
        Path to the local folder in which
        the file/s is/are located.

    Returns:
    --------
    tseries : dict
        Returns a dictionary with 'subjects_ids' as keys
        and BOLD time series as values.
    """ 
    try:
        try:
            tseries = load_dictionary(f'{path}/time_series.pkl')
        except:
            signals_path = f'{path}/time_series'
            sub_folders = [f for f in os.listdir(signals_path) if os.path.isdir(f'{signals_path}/{f}')]
            sub_folders.sort()

            tseries = {}

            for sub in sub_folders:
                for file in os.listdir(f'{signals_path}/{sub}'):
                    if file.endswith('.csv'):
                        tseries[sub] = np.array(
                                            pd.read_csv(
                                                f'{signals_path}/{sub}/{file}',
                                                sep=',',
                                                header=None
                                                )
                                            )
            
        return tseries
    except:
        raise Exception("The time series couldn't be loaded.")
    
def load_rois_coordinates(path):
    """
    Load the .csv file that contains
    the ROIs coordinates in MNI space
    and return it as a numpy array (N_ROIs,3)

    Params:
    -------
    path : str
        Path to the local folder
        in which the file is located.

    Returns:
    --------
    coords : ndarray with shape (n_rois,3) | None
        Returns a 2D array with the coordinates of each
        ROI/parcel if the file was succesfully loaded. 
        Otherwise returns None.
    """ 
    try:
        coords = pd.read_csv(f'{path}/rois_coordinates.csv',sep=',').values
        if coords.shape[1] != 3:
            print(f"Warning: the provided file with coordinates contain {coords.shape[1]} columns. "
                "Remember that this file must have a shape of (N_rois,3).")
            return None
        else:
            return coords
    except:
        return None