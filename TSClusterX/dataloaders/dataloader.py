class DataLoaderFactory:
    @staticmethod
    def get_dataloader(dataset_name, dataset_path):
        if dataset_name == 'ucr_uea':
            from .ucr_uea_dataloader import UCRClusterDataLoader
            return UCRClusterDataLoader(dataset_name, dataset_path)
        else:
            from .manual_dataloader import ManualTSDataLoader
            return ManualTSDataLoader(dataset_name, dataset_path)