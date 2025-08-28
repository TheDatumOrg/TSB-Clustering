class DistanceMeasure:
    def compute(self, series_set):
        """Compute NxN distance matrix for the given set of time series."""
        raise NotImplementedError("Subclasses should implement this!")



class DistanceFactory:
    @staticmethod
    def get_distance(name):
        if name is None:
            return None
        elif name == "euclidean":
            from distances.euclidean import EuclideanDistance
            return EuclideanDistance()
        elif name == "dtw":
            from distances.dtw import DTWDistance
            return DTWDistance()
        elif name == "edr":
            from distances.edr import EDRDistance
            return EDRDistance()
        elif name == "erp":
            from distances.erp import ERPDistance
            return ERPDistance()
        elif name == "gak":
            from distances.gak import GAKDistance
            return GAKDistance()
        elif name == "grail":
            from distances.grail import GRAILDistance
            return GRAILDistance()
        elif name == "kdtw":
            from distances.kdtw import kDTWDistance
            return kDTWDistance()
        elif name == "lcss":
            from distances.lcss import LCSSistance
            return LCSSistance()
        elif name == "msm":
            from distances.msm import MSMDistance
            return MSMDistance()
        elif name == "rbf":
            from distances.rbf import RBFDistance
            return RBFDistance()
        elif name == "sbd":
            from distances.sbd import SBDDistance
            return SBDDistance()
        elif name == "sink":
            from distances.sink import SINKDistance
            return SINKDistance()
        elif name == "swale":
            from distances.swale import SWALEDistance
            return SWALEDistance()
        elif name == "twed":
            from distances.twed import TWEDDistance
            return TWEDDistance()
        else:
            raise ValueError(f"Unknown distance type: {name}")
