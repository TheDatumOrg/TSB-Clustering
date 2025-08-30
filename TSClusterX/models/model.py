import numpy as np


class BaseClusterModel:
    def __init__(self, n_clusters, params=None, distance_name=None, distance_matrix=None):
        self.n_clusters = n_clusters
        self.params = params if params is not None else {}
        self.distance_name = distance_name
        self.distance_matrix = distance_matrix

    def fit_predict(self, X):
        raise NotImplementedError("Subclasses should implement this method.")


class ModelFactory:
    @staticmethod
    def get_model(model_name, n_clusters, params=None, distance_name=None, distance_matrix=None):
        if model_name == 'agglomerative':
            from models import agglomerative 
            return agglomerative.AgglomerativeClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'kmeans':
            from models import kmeans
            return kmeans.KMeansClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'densitypeaks':
            from models import densitypeaks
            return densitypeaks.DensityPeaksClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'pam':
            from models import pam
            return pam.PAMClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'dbscan':
            from models import dbscan
            return dbscan.DBSCANClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'birch':
            from models import birch
            return birch.BirchClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'affinitypropagation':
            from models import affinitypropagation
            return affinitypropagation.AffinityPropagationClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'gaussianmixture':
            from models import gaussianmixture
            return gaussianmixture.GaussianMixtureClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'spectralclustering':
            from models import spectralclustering
            return spectralclustering.SpectralClusteringModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'optics':
            from models import optics
            return optics.OPTICSClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'kdba':
            from models import kdba
            return kdba.KDBAClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'kernelkavg':
            from models import kernelkavg
            return kernelkavg.KernelKAvgClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'ksc':
            from models import ksc
            return ksc.KSCClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'catch22':
            from models import clustcatch22
            return clustcatch22.Catch22ClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'lpcc':
            from models import lpcc
            return lpcc.LPCCClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'ar_coeff':
            from models import ar_coeff
            return ar_coeff.ARCoeffClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'ar_pval':
            from models import ar_pval
            return ar_pval.ARPValClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'es_coeff':
            from models import es_coeff
            return es_coeff.ESCoeffClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'featts':
            from models import featts
            return featts.FeatTSClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'kasba':
            from models import kasba
            return kasba.KASBAClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'kshape':
            from models import kshape
            return kshape.KShapeClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'ushapelets':
            from models import ushapelets
            return ushapelets.UShapeletsClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'chronos':
            from models import chronos
            return chronos.ChronosClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'moment':
            from models import moment
            return moment.MomentClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'ofa':
            from models import ofa
            return ofa.OFAClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'dec':
            from models import dec
            return dec.DECClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'idec':
            from models import idec
            return idec.IDECClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'depict':
            from models import depict
            return depict.DEPICTClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'sdcn':
            from models import sdcn
            return sdcn.SDCNClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'dtc':
            from models import dtc
            return dtc.DTCClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'vade':
            from models import vade
            return vade.VADEClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'dtcr':
            from models import dtcr
            return dtcr.DTCRClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'som_vae':
            from models import som_vae
            return som_vae.SOMVAEClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'dcn':
            from models import dcn
            return dcn.DCNClusterModel(n_clusters, params, distance_name, distance_matrix)
        elif model_name == 'clustergan':
            from models import clustergan
            return clustergan.ClusterGANClusterModel(n_clusters, params, distance_name, distance_matrix)
        else:
            raise ValueError(f"Unknown model: {model_name}")
