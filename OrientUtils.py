#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:33:44 2020

@author: cat
what I used to make some 3D pictures
"""

import matplotlib.pyplot as plt
import nibabel as nib
from skspatial.objects import Points, Plane, Line
import numpy as np
import pandas as pd
# from skspatial.plotting import plot_3d
from scipy import optimize
from scipy.ndimage.interpolation import zoom
import os, pickle, copy, multiprocessing, researchpy, random, tqdm
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib
from matplotlib import pyplot as plt, font_manager as fm
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

IDs = {'sham': [1017, 1035, 1045, 1085, 1091, 1096, 1107, 1143, 1146, 1155, 1161], 
       'extrasham':[1007, 1026], # bad EEG
       'TBI-': [1019, 1028, 1036, 1038, 1043, 1046, 1090, 1095, 1099, 1103, 1105, 1138, 1139, 1142, 1145, 1149, 1150,
                1152, 1153, 1154, 1156, 1158, 1159],
       'TBI+': [1008, 1012, 1024, 1029, 1031, 1084, 1104, 1140, 1144],
       'TBI':[1027,1044,1081,1082,1086,1087,1092,1093,1097,1010,1018]} # not epylepsy phenotyped





ET_TBIPlus = ['12', '26', '32', '33', '42', '43', '57', '62', '68', '71', '74', '103', '107',
              '147', '148', '149', '151', '153', '155', '163', '171', '183', '184', '192',
              '199', '200', '205', '207', '220', '241']

ET_Exclude = ['62']

ET_SHAM = ['9', '17', '29', '39', '48', '60', '66', '78', '90', '101', '114', '125', '129',
           '146', '158', '165', '178', '186', '201', '212', '222', '242', '254']

extrasham = ['233']

extraTBI = ['253','88' ,'143','176' ,'218', '250' ]

ET_TBIMinus = ['2', '3', '4', '7', '8', '13', '15', '18', '20', '21', '22', '23', '24', '36',
               '37', '40', '41', '4402', '49', '51', '53', '54', '59', '65', '70', '73',
               '75', '80', '81', '83', '84', '86', '87', '94', '95', '97', '102', '105',
               '109', '112', '113', '116', '119', '120', '122', '124', '127', '132',
               '133', '134', '135', '137', '142', '157', '159', '160', '162', '167',
               '170', '172', '173', '182', '185', '187', '197', '198', '203', '204',
               '210', '214', '219', '221', '225', '226', '227', '229', '230', '235',
               '236', '238', '240', '245', '246', '248', '253']

ET_Naive = ['67', '79', '104', '136', '179', '189', '190', '213', '223', '224',
            '234', '243', '255']

LabelDict = {'sham': 0,
             'TBI+': 1,
             'TBI-': 0,
             'TBI' : 1,
             'TBI Naive': -2}

TimeDict = {'30d': 30,
            '2d': 2,
            '5mo': 5 * 30,
            '9d': 9,
            '2': 2,
            '7': 7,
            '21': 21}


def GetETLesionType(ID):
    # if ID in ET_Exclude: return False
    if ID in ET_SHAM:
        return 'sham'
    elif ID in ET_TBIPlus:
        return 'TBI+'
    elif ID in ET_TBIMinus:
        return 'TBI-'
    elif ID in ET_Naive:
        print('TBI Naive')
        return 'TBINaive'
    elif ID in extrasham:
        return 'extrasham'
    elif ID in extraTBI:
        return 'TBI'
    else:
        return False


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean()
    if std is None:
        std = X.std()
    return (X - mean) / std, mean, std


def GetLesionType(ID):
    """
    Given ID, returns lesion type: sham, TBI+., TBI-
    """
    for lesion, animals in IDs.items():
        if int(ID) in animals: return lesion
    # print(ID)
    return False


def MakePlane(vol): # P1
    """
    

    Parameters
    ----------
    vol : 3D numpy array
        Hippocampus binary mask.

    Returns
    -------
    plane : scikit-spatial plane object
        Check the plane.vector and plane.point objects for the parameters.

    """

    points = Points(np.argwhere(vol == 1))
    plane = Plane.best_fit(points)
    
    if plane.normal[1]>0:
        plane = Plane(plane.point,-np.array(plane.normal))

    return plane


def MakePlaneRef(rvol, refplane):
    """
    Returns transformation matrix for new reference based on plane
    Different vectors on axis 0: M[1,:] is E1
    Transform is n=np.dot(M.T,oldn)
    """
    RP = refplane
    allvecs = np.argwhere(rvol)
    p = allvecs
    centerpoint = RP.point
    # centerpointN=centerpoint/np.linalg.norm(centerpoint)

    # a and b are close, c is far
    md = np.inf
    for i in range(3):
        for j in range(3):
            d = np.linalg.norm(p[i] - p[j])
            if d < md and i != j:
                md = d
                a = i
                b = j
    for c in range(3):
        if c not in [a, b]: break
    # Orthonormal base:
    E1 = centerpoint - p[c]
    # same direction of Y, positive towards front of rat
    E1 = E1 / np.linalg.norm(E1)
    E2 = np.array(RP.vector)
    # Plane normal, same direction of Z
    if E2[2] < 0: E2 = -E2
    E2 = E2 / np.linalg.norm(E2)
    # Remaining axis from the cross product
    E0 = np.cross(E1, E2)

    return np.array([E0, E1, E2])


class GetNormPlane(Plane): # P2
    """
    P2
    """

    def __init__(self, vol,simpler = False):
        skeleton = skeletonize(vol).astype(float) / 255  # abs space

        allvecs = np.argwhere(vol == 1)  # abs space
        masscenter = np.mean(allvecs, axis=0)  # abs space

        # centeredvecs=allvecs-masscenter

        iniziale, finale = Extremes(skeleton)  # - masscenter # abs space

        midpoint = (finale + iniziale) / 2  # abs space

        CMdir = masscenter - midpoint  # centered
        CMdir = CMdir / np.linalg.norm(CMdir) # v1

        axis = finale - midpoint  # centered
        axis = axis / np.linalg.norm(axis) # v2

        onplane = np.cross(axis, CMdir)
        onplane = onplane / np.linalg.norm(onplane) # v3

        normal = np.cross(axis, onplane) # CMdir sucks here!
        normal = normal / np.linalg.norm(normal)
        # if normal[1] > 0:
        #     normal = -normal

        super(GetNormPlane, self).__init__(midpoint, normal)


def volfromcoords(x, y, z, shape, axis):
    a = zip(x.ravel(), y.ravel(), z.ravel())

    valid = np.array([el for el in a if el[axis] < shape[axis] and el[axis] >= 0])
    if len(valid) == 0: return 0
    valid = valid[:, 0], valid[:, 1], valid[:, 2]

    V = np.zeros(shape, dtype=int)

    V[valid] = 1

    return V


def planify(x, y, z, brainmask, center, norm):
    if np.any(norm == 0):
        norm = norm + EPS

    V = np.zeros_like(brainmask)

    x1, y1 = np.meshgrid(x, y, indexing='ij')
    Z = center[2] - (norm[0] * (x1 - center[0]) + norm[1] * (y1 - center[1])) / norm[2]
    Z = np.round(Z).astype(int)

    V += volfromcoords(x1, y1, Z, brainmask.shape, 2)

    y1, z1 = np.meshgrid(y, z, indexing='ij')
    X = center[0] - (norm[2] * (z1 - center[2]) + norm[1] * (y1 - center[1])) / norm[0]
    X = np.round(X).astype(int)

    V += volfromcoords(X, y1, z1, brainmask.shape, 0)

    x1, z1 = np.meshgrid(x, z, indexing='ij')
    Y = center[1] - (norm[2] * (z1 - center[2]) + norm[0] * (x1 - center[0])) / norm[1]
    Y = np.round(Y).astype(int)

    V += volfromcoords(x1, Y, z1, brainmask.shape, 1)

    V[V > 0] = 1
    return V


EPS = 1e-10


def mainvector(brainmask, plane):
    # Y axis of reference
    x = np.arange(brainmask.shape[0])
    y = np.arange(brainmask.shape[1])
    z = np.arange(brainmask.shape[2])

    point = np.array(plane.point)
    normal = np.array(plane.normal)

    Pvol = planify(x, y, z, brainmask, point, normal)
    allvecs = np.argwhere(Pvol == 1)

    X0 = allvecs[0] - point
    n0 = X0 / np.linalg.norm(X0)

    def pointy(n):
        line = Line(point, n)

        line.vector = line.vector / np.linalg.norm(line.vector)

        factor = np.dot((line.point - allvecs), line.vector)
        factor = np.stack((factor, factor, factor))

        d = np.linalg.norm((line.point - allvecs) - factor.T * line.vector, axis=1)

        return d.mean()

    def cons_f(norm):
        return np.dot(norm, norm)

    def cons_J(norm):
        return norm * 2

    def cons_H(norm, v):
        return v * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    cons = optimize.NonlinearConstraint(cons_f, 1, 1, jac=cons_J, hess=cons_H)
    res = optimize.minimize(pointy, n0, method='trust-constr',
                            options={'disp': False},
                            constraints=cons)

    n = res.x
    if n[1] < 0:
        n = -n

    return n


def maskref(brainmask):
    # makes brain mask reference frame
    refplane = FitBMRef(brainmask)
    y = mainvector(brainmask, refplane)
    z = np.array(refplane.vector)
    x = np.cross(y, z)

    return np.array([x, y, z]), refplane


class FitBMRef(Plane):
    """
    Fit a plane to a volume, looking for best fit
    """

    def __init__(self, brainmask):
        allvecs = np.argwhere(brainmask == 1)
        centerpoint = np.mean(allvecs, axis=0)

        norm0 = np.array([0.0001, 0.0001, 1])
        norm0 = norm0 / np.linalg.norm(norm0)
        x = np.arange(brainmask.shape[0])
        y = np.arange(brainmask.shape[1])
        z = np.arange(brainmask.shape[2])

        init = np.concatenate((norm0, centerpoint))

        def intersection(args):
            norm = args[:3]
            center = args[3:]

            V = planify(x, y, z, brainmask, center, norm)

            intersection = -np.sum(V * brainmask)
            return intersection

        boundaries = optimize.Bounds([-0.5, -0.5, 0.5, 0, 0, 0], [0.5, 0.5, 1] + list(brainmask.shape))

        def cons_f(norm):
            return np.dot(norm[:3], norm[:3])

        def cons_J(norm):
            inp = np.concatenate((norm[:3], (0, 0, 0)))
            return inp * 2

        def cons_H(norm, v):
            return v * np.array(
                [[2, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]])

        cons = optimize.NonlinearConstraint(cons_f, 1, 1, jac=cons_J, hess=cons_H)

        res = optimize.minimize(intersection, init, method='trust-constr',
                                options={'disp': False},
                                constraints=cons, bounds=boundaries)
        centerpoint = res.x[3:]
        norm = res.x[:3]
        if norm[2] < 0:
            norm = -norm
        super(FitBMRef, self).__init__(centerpoint, norm)


def Extremes(skeleton):
    """
    Given a skeleton, returns the two extremes

    Parameters
    ----------
    skeleton : np volume
        A numpy volume generated by scipy.morphology.skeletonize
        May or may not be normalized

    Returns
    -------
    initial : [x,y,z] np.array
        initial point of skeleton
    final : [x,y,z] np.array
        final point of the skeleton

    """
    points = np.argwhere(skeleton > 0)
    mid = np.mean(points, axis=0)
    upside = points[np.where(points[:, 2] < mid[2])]
    MV = np.argmax(np.linalg.norm(upside - mid, axis=1))
    final = upside[MV]

    downside = points[np.where(points[:, 2] > mid[2])]
    MV = np.argmax(np.linalg.norm(downside - mid, axis=1))
    initial = downside[MV]
    return initial, final


class PlaneFrom3(Plane):
    """
    A simple script to define a plane from 3 points
    """

    def __init__(self, refvol):
        allvecs = np.argwhere(refvol == 1)
        centerpoint = np.mean(allvecs, axis=0)
        vv = allvecs - centerpoint
        n = np.cross(vv[1] - vv[0], vv[2] - vv[0])
        n = n / np.linalg.norm(n)

        if n[2] < 0: n = -n

        super(PlaneFrom3, self).__init__(centerpoint, n)


def Dihedral(P1, P2):
    """
    

    Parameters
    ----------
    P1 : plane object
    P2 : plane object

    Returns
    -------
    Angle between planes, in radians

    """
    return np.arccos(np.dot(P1.normal, P2.normal))


def AddVector(dictionary, key, vec):
    """
    Adds a vector to the dictionary, as keyX, keyY, keyZ

    Parameters
    ----------
    dictionary : dict
        dictionary to add the entry to
    key : str
        key base name
    vec : TYPE
        np.array of lenght 3

    Returns
    -------
    None.

    """
    assert len(vec) == 3
    dictionary[key + 'X'] = vec[0]
    dictionary[key + 'Y'] = vec[1]
    dictionary[key + 'Z'] = vec[2]


def Featurize(ref, ipsi, contra, mask,
              voxshape=np.array((0.16, 0.16, 0.16)), flip=False,
              BMRef=True, mri=None):
    """
    Given paths of volumes, generate one data entry for animal

    Parameters
    ----------
    ref : str
        path to reference volume, 3 anatomical points
    ipsi : str
        ipsilateral hippocampus segmentation
    contra : str
        contralateral hippocampus segmentation
    voxshape : TYPE, np.array of floats
        Actual size of the voxels in each direction.
        The default is np.array((0.16,0.16,0.16)).

    Returns
    -------
    allfeatures : dict
        Feature dictionary for one sample.

    """

    allfeatures = {}
    voxsize = np.prod(voxshape)
    if BMRef:
        brainmask = nib.load(mask).get_fdata()
        bmvol = np.sum(brainmask)
        if flip: brainmask = np.swapaxes(brainmask, 2, 1)
        M, RP = maskref(brainmask)
    else:
        vref = nib.load(ref).get_fdata()
        if flip: vref = np.swapaxes(vref, 2, 1)
        RP = PlaneFrom3(vref)
        bmvol = 1
        M = MakePlaneRef(vref, RP)
    
    vipsi = nib.load(ipsi).get_fdata()
    vcontra = nib.load(contra).get_fdata()

    if flip:
        vipsi = np.swapaxes(vipsi, 2, 1)
        vcontra = np.swapaxes(vcontra, 2, 1)

    IP = MakePlane(vipsi)
    CP = MakePlane(vcontra)
    if IP.vector[0] < 0: IP.vector = -IP.vector
    if CP.vector[0] > 0: CP.vector = -CP.vector

    IN = GetNormPlane(vipsi)
    CN = GetNormPlane(vcontra)

    if mri is not None:
        MRI = nib.load(mri).get_fdata()
        MRI, _, _ = normalize(MRI)

        T2ipsi = np.mean(MRI[vipsi == 1])
        T2contra = np.mean(MRI[vcontra == 1])
        allfeatures['Ipsi mean T2'] = T2ipsi
        allfeatures['Contra mean T2'] = T2contra

    for L, P, V, N in zip(['Ipsi ', 'Contra '], [IP, CP], [vipsi, vcontra], [IN, CN]):
        allfeatures[L + 'RelVolume'] = np.sum(V) / np.sum(brainmask)
        AddVector(allfeatures, L + 'P1 Position', np.dot(M.T, P.point - RP.point) * voxshape)
        AddVector(allfeatures, L + 'P1 Normal', np.dot(M.T, P.vector))

        AddVector(allfeatures, L + 'P2 Position', np.dot(M.T, N.point - RP.point) * voxshape)
        allfeatures[L + 'P2 Distance'] = np.linalg.norm(np.dot(M.T, N.point - RP.point) * voxshape)
        allfeatures[L + 'P1 Distance'] = np.linalg.norm(np.dot(M.T, P.point - RP.point) * voxshape)
        AddVector(allfeatures, L + 'P2 Normal', np.dot(M.T, N.vector))

        allfeatures[L + 'P1 Angle'] = Dihedral(P, RP)
        allfeatures[L + 'P2 Angle'] = Dihedral(N, RP)
        
    allfeatures['Con-Ipsi P1 Angle'] = Dihedral(CP, IP)
    allfeatures['Con-Ipsi P2 Angle'] = Dihedral(CP, IN)
    allfeatures['Ipsi Contra Vol Ratio'] = np.sum(vipsi) / np.sum(vcontra)
    allfeatures['Ipsi P1 P2 Angle'] = Dihedral(IP, IN)
    allfeatures['Contra P1 P2 Angle'] = Dihedral(CP, CN)

    return allfeatures

def tryrefs(folder):
    """
    Names are inconsistent. Given a path, tries out other possible options

    Parameters
    ----------
    folder : str
        path to folder containing the volume

    Raises
    ------
    FileNotFoundError
        if no fine is found

    Returns
    -------
    File path: str
        path to the volume, if found

    """
    names = ['_mask_bone_coord.nii',
             '_mask_bone_coord.roi.nii']
    for name in names:
        ref = os.path.join(folder, 'MGRE_anatomy_' + folder.name + name)
        # print(ref)
        if os.path.isfile(ref): return ref
    raise FileNotFoundError


def ListData(path, respath):
    """
    lists all data given data path and reference volume folder
    """
    mice = []
    for folder in os.scandir(path):
        if os.path.isdir(folder):
            ID, timepoint = folder.name.split('_')
            ref = tryrefs(folder)

            ipsi = os.path.join(respath, folder.name + '_Ipsi.nii.gz')
            contra = os.path.join(respath, folder.name + '_Contra.nii.gz')

            mice.append({'ID': ID,
                         'ref': ref,
                         'ipsi': ipsi,
                         'contra': contra,
                         'timepoint': timepoint,
                         'lesion': GetLesionType(ID)
                         })
    return mice


from pathlib import Path


def LookForCFolder(filename, base='/media/Olowoo/Work/hippocampus'):
    for fold in ['C1', 'C2', 'C3']:
        for file in os.scandir(os.path.join(base, fold)):
            if filename.lower() in file.name.lower():
                return file.path
    print('File', filename, 'not found')
    return False


def ListEBData(respath, refpath, test=True):
    mice = []

    for F in os.scandir(respath):
        if 'Ipsi' in F.name and 'breathingMovement' not in F.name:
            basepath = str(Path(F.path).parent)

            ID, timepoint, _ = F.name.split('_')
            lestype = GetLesionType(ID)
            if lestype is not False:
                ipsi = os.path.join(basepath, ID + '_' + timepoint + '_Ipsi.nii.gz')
                contra = os.path.join(basepath, ID + '_' + timepoint + '_Contra.nii.gz')
                ref = os.path.join(refpath, 'MGRE_anatomy_' + ID + '_' + timepoint + '_mask_bone_coord.nii')
                bm = os.path.join(basepath, ID + '_' + timepoint + '_Mask.nii.gz')
                if not os.path.isfile(ref):
                    ref = os.path.join(refpath, 'MGRE_Anatomy_' + ID + '_' + timepoint + '_mask_bone_coord.nii')
                if test:

                    assert os.path.isfile(ipsi), ipsi
                    assert os.path.isfile(ref), ref
                    assert os.path.isfile(contra), contra
                    summy = nib.load(ref).get_fdata().sum()
                    if summy != 3:
                        print('WRONG SUM', ref, summy)

                mice.append({'ID': ID,
                             'ref': ref,
                             'ipsi': ipsi,
                             'contra': contra,
                             'MRI': LookForCFolder('MGRE_Anatomy_' + ID + '_' + timepoint + '.nii'),
                             'timepoint': timepoint,
                             'lesion': lestype,
                             'brainmask': bm
                             })
    return mice


def GetETVolums(name, basepath='/media/Olowoo/Work/hippocampus/epitarget_all'):
    for root, dirs, files in os.walk(basepath):
        if name in root:
            return os.path.join(root, 't2star_sumOverEchoes.nii')
    print('File', name, 'not found')
    return False


def ListETData(respath, test=True):
    mice = []
    for F in os.scandir(respath):
        if 'Ipsi' in F.name:
            basepath = str(Path(F.path).parent)

            basename, _ = F.name.rsplit('_', 1)
            ID, timepoint = basename.rsplit('DAY')
            ID = ID.rstrip('_')
            ID = ID[:len(ID) - 8].replace('EPI_MRI_', '')
            
            lestype = GetETLesionType(ID)
            if lestype is not False:
                ipsi = os.path.join(basepath, basename + '_Ipsi.nii.gz')
                contra = os.path.join(basepath, basename + '_Contra.nii.gz')
                bm = os.path.join(basepath, basename + '_Mask.nii.gz')

                if test:
                    assert os.path.isfile(ipsi), ipsi
                    assert os.path.isfile(contra), contra
                mice.append({'ID': ID,
                             'ref': None,
                             'ipsi': ipsi,
                             'contra': contra,
                             'MRI': GetETVolums(basename[0]),
                             'timepoint': timepoint,
                             'lesion': lestype,
                             'brainmask': bm
                             })
    return mice


def MakeDataset(vols, voxshape=np.array((0.16, 0.16, 0.16)), flip=False,
                allvols=False,KeepID=False):
    """
    Goes through all the volumes, returns dataframe object

    Parameters
    ----------
    vols : str
        path to reference ovlumes folder
    respath : str
        path to segmentation results

    Returns
    -------
    pandas.DataFrame
        dataframe object for alla sata samples
def Featurize(ref,ipsi,contra,mask,
              voxshape=np.array((0.16,0.16,0.16)), flip=True,
    """

    alldatapoints = []
    P = multiprocessing.Pool(23)
    processes = {}

    for k in vols:
        processes[k['ipsi']] = P.apply_async(Featurize, (
        k['ref'], k['ipsi'], k['contra'], k['brainmask'], voxshape, flip, True, None))

    for k in vols:
        L = k['lesion']
        nope = False
        if L == 'extrasham':
            if allvols:
                L = 'sham'
            else:
                nope = True
        if L == 'TBI':
            if not allvols:
                nope = True
                
        # o=Featurize(k['ref'],k['ipsi'],k['contra'],k['brainmask'])
        # except:
        #     print('ERROR!')
        #     return k
        if not nope:
            try:
                o = processes[k['ipsi']].get()
                o['lesion'] = L
                o['timepoint'] = TimeDict[k['timepoint']]
                if KeepID: o['ID'] = k['ID']
                alldatapoints.append(
                    pd.DataFrame(o, index=[k['ID'] + '_' + k['timepoint']])
                )
            except:
                print('Item discarded')
                print(k)
                
    P.close()
    P.join()
    P.terminate()

    return pd.concat(alldatapoints)


def fd(d, key):
    if d is None: return None
    try:
        return d[key]
    except KeyError:
        return None
    
    
def NormalizeDataset(Dataframe, norms=None, keeplesions=True,
                     replacelesions=False, TBIonly=False):
    D = copy.deepcopy(Dataframe)
    nnorms = {}

    for r in ['Ipsi ', 'Contra ']:
        for k in ['RelVolume', 'P2 Distance','P1 Distance', 'P1 Angle', 'P2 Angle']:
            try:
                D[r + k], nnorms[r + k], nnorms[r + k + 'std'] = normalize(D[r + k], fd(norms, r + k),fd(norms, r + k + 'std'))
            except:
                pass

    D['timepoint'], nnorms['timepoint'], nnorms['timepoint' + 'std'] = normalize(D['timepoint'], fd(norms, 'timepoint'),
                                                                                 fd(norms, 'timepoint' + 'std'))
    
    for r in ['Ipsi ', 'Contra ']:
        for k in ['P1 Position', 'P1 Normal', 'P2 Position', 'P2 Normal']:
            for c in ['X', 'Y', 'Z']:
                try:
                    D[r + k + c], nnorms[r + k + c], nnorms[r + k + c + 'std'] = normalize(D[r + k + c],fd(norms, r + k + c),fd(norms, r + k + c + 'std'))
                except:
                    pass

    if TBIonly:
        for k in range(len(Dataframe['lesion'])):
            if 'TBI' in D['lesion'][k]:
                D['lesion'][k] = 'TBI'
    else:
        lesioned = ['TBI' in k for k in D['lesion']]
        D = D[lesioned]

    if replacelesions:
        BinarizeLesions(D)
    if not keeplesions:
        del D['lesion']
    return D, nnorms

def BinarizeLesions(D):
    for k in range(len(D['lesion'])):
            D['lesion'][k] = LabelDict[D['lesion'][k]]

def HasTBI(arr):
    a = []
    for k in arr:
        if type(k) == int:
            a.append(k == 1)
        else:
            a.append('TBI' in k)
    return a


def GetChance(labels, iters=1000):
    T = []
    labels = np.array(labels)
    uniques = []
    for k in labels:
        if k not in uniques:
            uniques.append(k)

    for k in range(iters):
        n = np.random.choice(labels, size=len(labels), replace=True)
        T.append(sum(labels == n) / len(labels))

    print('\nChance:', np.mean(T))

    for k in uniques:
        t = [k for j in range(len(labels))]
        eq = t == labels
        print(k, sum(eq) / len(eq))


from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import feature_selection
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
"""
SHAP structure, from outer to inner index:
    test fold
    output category index
    test set
    features
"""


def SortyIDX(xx):
    x = copy.deepcopy(xx)
    order = np.zeros_like(x)

    for k in range(len(x)):
        M = np.argmax(x)
        x[M] = -np.inf
        order[k] = int(M)
    return order


def Reorder(A, order):
    new = []
    for k in order:
        new.append(A[int(k)])
    return new


def OrderedPlot(scores, labels, figsize=(8, 11), savefile=None):
    order = SortyIDX(scores)

    X = Reorder(scores, order)
    Y = Reorder(labels, order)

    sns.barplot(x=X, y=Y)  # ,order=sorty(allcategories)
    # ax.fig.set_size_inches(15,14)
    sns.set(rc={'figure.figsize': figsize})
    if type(savefile) is str:
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()

polygrid = {'degree': np.arange(1, 5),
            'C': np.power(10, np.arange(2, 9)) / 100,
            'coef0': np.arange(-20, 71, step=2) / 10,
            'kernel': ['poly']}

# RBFgrid={'gamma':np.power(10,np.arange(0,7))/1000000,
#          'C':np.power(10,np.arange(0,9))/10000,
#          'kernel':['rbf']}
grids = [polygrid]

def EvalEffectSize(dataset, timepoint=None,
                   lesiononly=False, conflate_lesions=False, csvout=None):
    if timepoint is not None:
        D = dataset[dataset['timepoint'] == timepoint]
    else:
        D = dataset
    if conflate_lesions:
        L1 = 'TBI'
        L2 = 'sham'
    else:
        L1 = 'TBI+'
        L2 = 'TBI-'
    a1, b = NormalizeDataset(D, keeplesions=True, TBIonly=conflate_lesions)
    c1, _ = NormalizeDataset(D, b, TBIonly=conflate_lesions)
    lesioned = HasTBI(c1['lesion'])
    if timepoint is not None:
        a1['timepoint'] = 0
        c1['timepoint'] = 0

    if lesiononly:
        D = D[lesioned]
        a1 = a1[lesioned]
        c1 = c1[lesioned]

    results = {}

    for k in a1.keys():
        if k == 'lesion': continue
        g1 = a1[a1['lesion'] == L1][k]
        g2 = a1[a1['lesion'] == L2][k]

        des, res = researchpy.ttest(g1, g2, equal_variances=False)

        cohend = list(res.iloc[6])[1]

        welch = list(res.iloc[2])[1]

        pval = list(res.iloc[3])[1]

        difference = list(res.iloc[0])[1]

        results[k] = [cohend, difference, welch, pval]

    results = dict(sorted(results.items(), key=lambda item: np.abs(item[1][0]), reverse=True))

    if csvout is not None:
        csvout = open(csvout, 'w')

    print('Variable,Cohen\'s d,' + L1 + ' - ' + L2 + ',Welch t,p-val', file=csvout)
    for k, v in results.items():
        print(k, v[0], v[1], v[2], v[3], sep=',', file=csvout)

    return results


def UniqueTMPs(D):
    uniques = []
    for k in D['timepoint']:
        if k not in uniques:
            uniques.append(k)
    return uniques

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score,roc_curve
from collections import Counter
from sklearn.linear_model import LogisticRegression


def EvalRFC(dataset, timepoint=None, outerCV=10, innerCV=10, trees=1000, depth = 1,
            jobs=23, lesiononly=False, conflate_lesions=False, zeroshaps=False,
            shap_plot_name=None, BalAcc=True, bootstrap = False, verbose=False,
            BAonly=True,SplitRandomState = 1):
    
    if timepoint is not None:
        D = copy.deepcopy(dataset[dataset['timepoint'] == timepoint])
    else:
        D = copy.deepcopy(dataset)
    
    count = 0
    if bootstrap:
        while count < 10:
            p = random.choices(D['lesion'],k=len(D['lesion']))
            count = min(list(Counter(p).values()))
        D['lesion'] = p
    gini_importances = []
    perm_importances=[]
    
    outerkf = StratifiedKFold(n_splits=outerCV, shuffle=True,random_state=SplitRandomState)

    scores = []
    CMs = []
    forests = []
    shaps = []
    ForRocP = []


    predictions = []
    GT = []

    a1, b = NormalizeDataset(D, keeplesions=False, TBIonly=conflate_lesions,replacelesions=True)
    c1, _ = NormalizeDataset(D, b, TBIonly=conflate_lesions,replacelesions=True)
    
    if timepoint is not None:
        a1['timepoint'] = 0
        c1['timepoint'] = 0
            
    outerkf = StratifiedKFold(n_splits=outerCV, shuffle=True,random_state=SplitRandomState)
    
    for train, test in outerkf.split(c1, list(c1['lesion'])):
        c = c1.iloc[train]
        a = a1.iloc[train]
        
        forest = RandomForestClassifier(trees,random_state=None,
                                        class_weight='balanced_subsample',
                                        max_depth=depth,
                                        n_jobs=jobs,
                                        bootstrap = False)
        
        forest.fit(a,list(c['lesion']))
            
        forests.append( forest )

        t = a1.iloc[test]
        
        if timepoint is not None: t['timepoint'] = 0
        tc = c1.iloc[test]

        res = forest.predict(t)

        acc = tc['lesion'] == res
        
        predictions += list(res)
        GT += list(tc['lesion'])
        
        CM = confusion_matrix(list(tc['lesion']), res)
        CMs.append(CM)
        
        y_proba = forest.predict_proba(t)[:,1]
        ForRocP += list(y_proba)

        
        if verbose:
            importances = forest.feature_importances_
            feature_names = list(a.keys())
            forest_importances = pd.Series(importances, index=feature_names)
            gini_importances.append(forest_importances)
            
            pimp = permutation_importance(forest,t,list(tc['lesion']),n_jobs=jobs)
            perm_importances.append(pd.Series(pimp.importances_mean, index=feature_names))

        if zeroshaps:
            BG = np.zeros((1, a.shape[1]))
        else:
            BG = a

        

        scores.append(np.mean(acc))

    string = '\nAverage Accuracy: ' + str(np.mean(scores))
    ba = balanced_accuracy_score(GT, predictions)
    metrics = allscores(GT,predictions)
    
    if verbose:
        print(string)
        if timepoint is None:
            CMs = np.mean(np.array(CMs), axis=0)
            print(CMs)
        
        print('Balanced accuracy', ba)
    if BAonly: 
        return ba, metrics

    return scores, ba, forests, shaps, sum(gini_importances), sum(perm_importances), (GT,ForRocP,predictions), metrics

from sklearn.metrics import accuracy_score, f1_score

iterations = 10000
def testpvals(timepoint,dataset,dictentry,conflate=False,lesion_only=True, iters = iterations):
    
    P = multiprocessing.Pool(12)
    
    ProcessList=[]
    
    allBAtrue = ManyCVSplits(dataset, timepoint, shap_plot_name=name + str(timepoint) + '_TBI+_TBI-.png', lesiononly=lesion_only,bootstrap=False,jobs=None,
                          depth=1, conflate_lesions=conflate, SplitRandomStates = randstates,BAonly = False)
    bas = []
    dics = []
    for k in allBAtrue:
        bas.append(k[1])
        dics.append(k[7])
    
    
    allmetrics = {k:[] for k in dics[0]}
    for item in dics:
        for k in item:
            allmetrics[k] += [item[k]]
            
    allcomparisons = {k:[] for k in dics[0]}
    
    BAtrue = np.mean(bas)
    print('\nBalanced Accuracy:',np.round(BAtrue,4),flush=True)
    bootres = {'accuracies' : [],
               'maccuracies' : [],
               'original' : allBAtrue}
    
    
    for k in range(iters):
        ProcessList.append(P.apply_async(ManyCVSplits,(dataset, timepoint, 10, 10, 1000, 1,
                None, lesion_only, conflate, True,
                None, True, True, False,
                True, randstates)
                                                  ))
    k = 0
    with tqdm.tqdm(total=len(ProcessList)) as pbar:
        for process in ProcessList:
            F = process.get()
            a = []
            dicts = []
            for g in F:
                a.append(g[0])
                dicts.append(g[1])
            
            bootres['accuracies'].append(a)
            bootres['maccuracies'].append(np.mean(a))
            pbar.update()
            ratio = np.sum(k >= BAtrue for k in bootres['maccuracies'])/len(bootres['maccuracies'])
            pbar.set_description('p-val: '+str(np.round(ratio,4))+' latest BA: '+str(np.mean(np.round(a,4))))
            k += 1
            for item in dicts:
                for i in item:
                    allcomparisons[i] += [item[i]]
    pvals_for_all = {}
    for metric, comparisons in allcomparisons.items():
        pvals_for_all[metric] = np.sum(np.array(comparisons) > np.mean(allmetrics[metric]))/len(comparisons)
    pvals_for_all['data'] = allcomparisons
    P.close()
    P.join()
    P.terminate()
    del P
    
    pval = np.sum(k >= BAtrue for k in bootres['maccuracies'])/len(bootres['maccuracies'])
    print(flush=True)
    if pval == 0:
        print('p-value < ',1/len(bootres['maccuracies']))
    else:
        print('p-value = ',pval)
    bootres['pval'] = pval
    allres[dictentry] = pvals_for_all
    
    return pval


def allscores(true,pred):
    t = np.array(true)
    p = np.array(pred)
    tp = np.sum((p==1) * t)
    tn = np.sum((p==0) * (1-t))
    fp = np.sum((p==1) * (1-t))
    fn = np.sum((p==0) * t)
    
    AllMetrics = {'Accuracy':accuracy_score(true,pred),
                  'Balanced Accuracy':balanced_accuracy_score(true,pred),
                  'Sensitivity':tp/(tp + fn),
                  'Specificity':tn/(tn + fp),
                  'PPV':tp/(tp + fp),
                  'NPV':tn/(tn + fn),
                  'Precision':tp/(tp + fp),
                  'Recall':tp/(tp + fn),
                  'F1':f1_score(true,pred)}
    return AllMetrics

def ManyCVSplits(dataset, timepoint=None, outerCV=10, innerCV=10, trees=1000, depth = 1,
            jobs=23, lesiononly=False, conflate_lesions=False, zeroshaps=True,
            shap_plot_name=None, BalAcc=True, bootstrap = False, verbose=False,
            BAonly=False,SplitRandomStates = range(10)):
    
    balaccs = []

    NewD = copy.deepcopy(dataset)
    
    count = 0
    if bootstrap:
        while count < 10:
            p = random.choices(NewD['lesion'],k=len(NewD['lesion']))
            count = min(list(Counter(p).values()))
        NewD['lesion'] = p
    
    for seed in SplitRandomStates:
        balaccs.append(EvalRFC(NewD, timepoint, outerCV, innerCV, trees, depth,
                jobs, lesiononly, conflate_lesions, zeroshaps,
                shap_plot_name, BalAcc, False, verbose,
                BAonly,seed))
    return balaccs
import warnings

warnings.filterwarnings("ignore")

Epibios = True
remake = False

def MakeROC(allres,ax,auclab='',title='',filename = None):
    # (GT,ForRocP,predictions)
    
    GT = []
    score = []
    prediction = []
    
    for k, sample in enumerate(allres):
        gt, sc, pr = sample[6]
        GT += gt
        score += sc
        prediction += pr
        fpr, tpr, thresholds = roc_curve(gt, sc)
        auc = roc_auc_score(gt,sc)
        # plt.plot(fpr, tpr, linestyle=':') # 'RF '+ str(k)+
    
    fpr, tpr, thresholds = roc_curve(GT, score)
    auc = roc_auc_score(GT,score)
    
    auclab += 'AUC: '+str(np.round(auc,3))
    
    ax.plot(fpr, tpr,label=auclab)
    ax.set_title(title)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc='best')
    # if filename is not None: plt.savefig(filename,bbox_inches='tight')
    
def FixLabelTexts(obls,symbols=False):
    matplotlib.rc('xtick', labelsize=12) 
    # matplotlib.rc('ytick', labelsize=20) 
    labels = copy.deepcopy(obls)
    del labels['timepoint']
    if symbols:
        oldkeys = list(labels.keys())
        for k in oldkeys:
            s = ''
            if 'Ipsi' in k: s+='$Ipsi$ '
            if 'Contra' in k: s+='$Contra$ '
            if 'P1 Angle' in k:
                s += '$\\theta_{1,R}$'
            elif 'P2 Angle' in k:
                s += '$\\theta_{2,R}$'
            elif 'P1 P2 Angle' in k: 
                s += '$\\theta_{1,2}$'
            else:
                if 'P1' in k: s += '$p^{1}'
                if 'P2' in k: s += '$p^{2}'
                if k[-1] in ['X','Y','Z']:
                    component = k[-1].lower()+'}'
                else:
                    component = ''
                
                if 'Position' in k: 
                    s += '_{p'
                elif 'Normal' in k:
                    s += '_{n'
                elif 'Distance' in k:
                    # print(s)
                    s = s.replace('p^','|\\bf{p^')
                    s += '_{p}|'
                
                s += component
                if 'RelVolume' in k:
                    if 'Ipsi' in k:
                        s = '$v_{ipsi}'
                    else:
                        s = '$v_{contra}'
                s += '$'
                
            if k == 'Con-Ipsi P1 Angle':
                s = '$\\theta_{1,1}$'
            elif k == 'Con-Ipsi P2 Angle':
                s = '$\\theta_{2,2}$'
            
            if k == 'Ipsi Contra Vol Ratio':
                s = '$v_r$'
            # print('previous',k,'new',s)
            labels[s] = labels.pop(k)
    return labels


def ImportancePlot(manyruns,axgini,axperms,titlegini,titleperms,gini=False):
    ginis = []
    perms = []
    for k in (manyruns):
        ginis.append(k[4])
        perms.append(k[5])
    ginis = sum(ginis)/len(ginis)
    perms = sum(perms)/len(perms)
    
    ginis = FixLabelTexts(ginis)
    perms = FixLabelTexts(perms)
    
    headG = ginis.sort_values(ascending=False)#.head(20)
    headP = perms.sort_values(ascending=False)#.head(10)
    
    if gini:
        sns.barplot(headG.index,headG.values,ax = axgini)
        axgini.set_title(titlegini)
        axgini.set_ylabel('Gini importance')
        axgini.set_xticklabels(axgini.xaxis.get_majorticklabels(), Rotation = 45, ha = 'right')
    else:
        axperms = axgini
        sns.barplot(headP.index,headP.values, ax = axperms)
        axperms.set_title(titleperms)
        axperms.set_ylabel('Permutation importance')
        axperms.set_xticklabels(axperms.xaxis.get_majorticklabels(), Rotation = 45, ha = 'right')

def TopFeatures(dataset,manyruns,number=20):
    ginis = []
    newd = copy.deepcopy(dataset)
    for k in (manyruns):
        ginis.append(k[4])
    ginis = sum(ginis)/len(ginis)
    headG = ginis.sort_values(ascending=False).head(number)
    features = list(headG.keys()) + ['lesion','timepoint']
    
    for k in list(dataset.keys()):
        if k not in features: del newd[k]
        
    return newd

def RunsAcc(runs):
    accs = []
    for run in runs:
        accs += [run[1]]
    return np.mean(accs)

import pingouin as pg
excludedkeys = ['lesion','timepoint','ID']
def Anovas(dataset,TBIvsSham=True,filename=None):
    D = NormalizeDataset(copy.deepcopy(dataset), norms=None, keeplesions=True,replacelesions=False, TBIonly=TBIvsSham)[0]
    collected=[]
    for feature in list(D.keys()):
        if feature in excludedkeys: continue
        res = pg.rm_anova(D,feature,['timepoint','lesion'],'ID',detailed = True)
        res['Feature'] = [feature,'','']
        p1 = res['p-GG-corr'][0]*39
        p2 = res['p-GG-corr'][1]*39
        p3 = res['p-unc'][2]*39
        res['p-bonf-corr']=[p1,p2,p3]
        res = res.set_index('Feature')
        
        collected.append(res)
        
    collected = pd.concat(collected)
    if filename is not None: collected.to_csv(filename)
    return collected
        
# %%


if remake:
    vols = ListETData('/media/Olowoo/Work/hippocampus/resultsET')
    
    DET=MakeDataset(vols,voxshape=((0.15,0.15,0.5)), flip=True)
    pickle.dump(DET,open('ET_expanded_results.p','wb'))
    vols = ListEBData('/media/Olowoo/Work/hippocampus/results',
                      '/media/Olowoo/Work/hippocampus/bone_coord')
    D = MakeDataset(vols)
    pickle.dump(D,open('EB_expanded_results.p','wb'))
    DET.to_csv('ET_expanded_results.csv')
    D.to_csv('EB_expanded_results.csv')
else:
    
    D = pickle.load(open('EB_expanded_results.p', 'rb'))
    DET = pickle.load(open('ET_expanded_results.p', 'rb'))

name = 'EB'
if not Epibios:
    D = DET
    name = 'ET'

allBAtrue = {}
allBAtrue1 = {}
allBAtrueS = {}
allBAtrueS1 = {}
plt.rcParams.update({'font.size': 13})
conflate=False
lesion_only=True
randstates=range(10)

forpaperfig, forpaperax = plt.subplots(3,1,figsize=(12,5.3*3))

figure, axis = plt.subplots(1, 2,figsize=(12,5.3))
figure_imp, axis_imp = plt.subplots(4, 1,figsize=(12,5.3*4))

for k, timepoint in enumerate([2,9,30,150]):
    fname = None

    t1 = str(timepoint)+' days'
    t2 = str(timepoint)+' days'
    allBAtrue[timepoint] = ManyCVSplits(D, None, shap_plot_name=name + str(timepoint) + '_TBI+_TBI-.png', lesiononly=lesion_only,bootstrap=False,jobs=None,verbose=True,
                              depth=1, conflate_lesions=conflate, SplitRandomStates = randstates,BAonly = False)
    MakeROC(allBAtrue[timepoint],axis[0],str(timepoint) + ' d, ','EpiBioS4Rx TBI+ $vs.$ TBI-',fname)
    ImportancePlot(allBAtrue[timepoint],axis_imp[k],None,t1,t2)

figure_imp.subplots_adjust(hspace=0.5)
figure_imp.suptitle('EpiBioS4Rx TBI+ $vs.$ TBI- feature importance',y=.9857)
figure_imp.savefig('EBimportances.png',bbox_inches='tight')

figure150, axis150 = plt.subplots(1, 1,figsize=(12,5.3))
ImportancePlot(allBAtrue[150],axis150,None,'','')
figure150.suptitle('150 days TBI+ $vs.$ TBI- feature importance')
figure150.savefig('150EBimportance.png',bbox_inches='tight')

del figure_imp
del axis_imp
figure_imp, axis_imp = plt.subplots(3, 1,figsize=(12,5.3*3))


for k, timepoint in enumerate([2,7,21]):
    t1 = str(timepoint)+' days'
    t2 = str(timepoint)+' days'
    if timepoint == 21:
        fname = 'roc.eps'
    else:
        fname = None
    allBAtrue1[timepoint] = ManyCVSplits(DET, timepoint, shap_plot_name=name + str(timepoint) + '_TBI+_TBI-.png', lesiononly=lesion_only,bootstrap=False,jobs=None,verbose=True,
                              depth=1, conflate_lesions=conflate, SplitRandomStates = randstates,BAonly = False)
    MakeROC(allBAtrue1[timepoint],axis[1],str(timepoint) + ' d, ','EPITARGET TBI+ $vs.$ TBI-',fname)
    ImportancePlot(allBAtrue1[timepoint],axis_imp[k],None,t1,t2)

figure_imp.subplots_adjust(hspace=0.5)
axis[0].text(-.1,1.09,'C',fontsize = 18)
axis[1].text(-.1,1.09,'D',fontsize = 18)
figure.savefig('roc.png',bbox_inches='tight')
figure_imp.suptitle('EPITARGET TBI+ $vs.$ TBI- feature importance',y=.9857)
figure_imp.savefig('ETimportances.png',bbox_inches='tight')

conflate=True
lesion_only=False
randstates=range(10)

figure, axis = plt.subplots(1, 2,figsize=(12,5.3))
figure_imp, axis_imp = plt.subplots(4, 1,figsize=(12,5.3*4))

allBAtrueS1 = pickle.load(open('allBAtrueS1.p','rb'))#{}
allBAtrueS = pickle.load(open('allBAtrueS.p','rb'))#{}
for k, timepoint in enumerate([2,9,30,150]):
    fname = None

    t1 = str(timepoint)+' days'
    t2 = str(timepoint)+' days'
    allBAtrueS[timepoint] = ManyCVSplits(D, timepoint, shap_plot_name=name + str(timepoint) + '_TBI+_TBI-.png', lesiononly=lesion_only,bootstrap=False,jobs=None,verbose=True,
                               depth=1, conflate_lesions=conflate, SplitRandomStates = randstates,BAonly = False)
    MakeROC(allBAtrueS[timepoint],axis[0],str(timepoint) + ' d, ','EpiBioS4Rx TBI vs Sham',fname)
    ImportancePlot(allBAtrueS[timepoint],axis_imp[k],None,t1,t2)
ImportancePlot(allBAtrue[150],forpaperax[2],'150 days a','150 days b','TBI+ $vs.$ TBI- at 150 days',None)
ImportancePlot(allBAtrueS[2],forpaperax[0],'2 days','2 days','TBI $vs.$ Sham at 2 days',None)
ImportancePlot(allBAtrueS[150],forpaperax[1],'150 days','150 days','TBI $vs.$ Sham at 150 days',None)

forpaperax[0].text(-3,.13,'A',fontsize=18)
forpaperax[1].text(-3,.8,'B',fontsize=18)
forpaperax[2].text(-3,.75,'C',fontsize=18)

figure_imp.subplots_adjust(hspace=0.5)
figure_imp.suptitle('EpiBioS4Rx TBI $vs.$ Sham feature importance',y=.9857)
figure_imp.savefig('ShamEBimportances.png',bbox_inches='tight')
forpaperfig.savefig('forpaper.png')

figure150, axis150 = plt.subplots(2, 1,figsize=(12,5.3*2))
ImportancePlot(allBAtrueS[150],axis150[1],None,'A','150 days')
ImportancePlot(allBAtrueS[2],axis150[0],None,'Gini importance','2 days')
figure150.subplots_adjust(hspace=0.5)
figure150.suptitle('TBI $vs.$ Sham feature importance')
figure150.savefig('Sham150EBimportance.png',bbox_inches='tight')


del figure_imp
del axis_imp
figure_imp, axis_imp = plt.subplots(3, 1,figsize=(12,5.3*3))


for k, timepoint in enumerate([2,7,21]):
    t1 = str(timepoint)+' days'
    t2 = str(timepoint)+' days'
    if timepoint == 21:
        fname = 'roc.eps'
    else:
        fname = None
    allBAtrueS1[timepoint] = ManyCVSplits(DET, timepoint, shap_plot_name=name + str(timepoint) + '_TBI+_TBI-.png', lesiononly=lesion_only,bootstrap=False,jobs=None,verbose=True,
                              depth=1, conflate_lesions=conflate, SplitRandomStates = randstates,BAonly = False)
    MakeROC(allBAtrueS1[timepoint],axis[1],str(timepoint) + ' d, ','EPITARGET TBI vs Sham',fname)
    ImportancePlot(allBAtrueS1[timepoint],axis_imp[k],None,t1,t2)

figure_imp.subplots_adjust(hspace=0.5)

axis[0].text(-.1,1.09,'A',fontsize = 18)
axis[1].text(-.1,1.09,'B',fontsize = 18)

figure.savefig('Shamroc.png',bbox_inches='tight')
figure_imp.suptitle('EPITARGET TBI $vs.$ Sham feature importance')
figure_imp.savefig('ShamETimportances.png',bbox_inches='tight')

# pickle.dump(allBAtrueS1,open('allBAtrueS1.p','wb'))
# pickle.dump(allBAtrueS,open('allBAtrueS.p','wb'))
#%% bootstrap test for results p-value
allres={}

# for conf, les in zip([False,True],[True,False]):
#     if conf:
#         exp = ' TBI vs Sham'
        
#     else:
#         exp = ' TBI+ vs TBI-'
    
#     print(exp)

#     for time in [2,9,30,150, None]:
#         print('EpiBioS',time)
#         testpvals(time,D,'EB '+str(time)+exp,conf,les)
    
#     for time in [2,7,21, None]:
#         print('EpiTARGET',time)
#         testpvals(time,DET,'ET '+str(time)+exp,conf,les)
    
# pickle.dump(allres,open(fn,'wb'))
