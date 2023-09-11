import os
from dataclasses import dataclass
from typing import Optional, Dict, Callable
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, MDS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn . decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from cebra import CEBRA

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../.."))


@dataclass
class DayEO:
    # different datasets
    binocular: Optional[np.ndarray]

    contra: Optional[np.ndarray]

    ipsi: Optional[np.ndarray]

    spont: Optional[np.ndarray]


class Dataset:
    def __init__(self):
        """
        Datashape:
        [time in the trial, trial, stimuli, X-dimension, Y-Dimension]
        """
        self.data = self.load()

    @staticmethod
    def load() -> Dict[str, DayEO]:
        days = [os.path.join(BP, "data/Share_TN2", d) for d in
                os.listdir(os.path.join(BP, "data/Share_TN2"))]
        data = dict()
        for day in days:
            prefix = day.split("/")[-1]
            data[prefix] = DayEO(None, None, None, None)
            temp_files = [os.path.join(day, f) for f in os.listdir(day)]
            for temp_file in temp_files:
                if "binocular" in temp_file:
                    data[prefix].binocular = np.load(temp_file)
                elif "contra" in temp_file:
                    data[prefix].contra = np.load(temp_file)
                elif "ipsi" in temp_file:
                    data[prefix].ipsi = np.load(temp_file)
                elif "spont" in temp_file:
                    data[prefix].spont = np.load(temp_file)
        return data

    def verify_dataset(self):
        print(f"Days: {self.data.keys()}")
        for day in self.data.keys():
            print(20*"=")
            print(f"{day}:")
            print(f"binocular-shape: {self.data[day].binocular.shape}")
            print(f"contra-shape: {self.data[day].contra.shape}")
            print(f"ipsi-shape: {self.data[day].ipsi.shape}")
            print(f"spont-shape: {self.data[day].spont.shape}")
            print(20*"=")

    def task_1(self,
               reduction_method: Callable[[np.ndarray], np.ndarray],
               day: str = "Day_EO+6",
               time_point: int = 2
               ):
        orig_shape = self.data[day].binocular.shape
        print(orig_shape)
        d1 = self.data[day].binocular
        # preprocess
        d1 = np.reshape(d1, d1.shape[:3]+tuple([d1.shape[-1] * d1.shape[-2]]))
        print(d1.shape)
        d1 = np.reshape(d1[time_point], (d1[time_point].shape[0] * d1[time_point].shape[1], d1[time_point].shape[-1]))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', keep_empty_features=True, fill_value=0)
        d1 = imp_mean.fit_transform(d1)
        x = preprocessing.normalize(d1)

        # Dim-Reduction:
        reduced_data = reduction_method(x)
        colors = []
        """
        for i in range(orig_shape[1]):
        # for j in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
            for j in [0, 45, 90, 135, 22.5, 67.5, 112.5, 157.5]:
                colors.append(j)"""
        for i in range(orig_shape[1] // 2):

            # for j in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
            for j in [0, 45, 90, 135, 0, 45, 90, 135]:
                colors.append(j)
        for i in range(orig_shape[1] // 2, orig_shape[1]):

            # for j in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
            for j in [22.5, 67.5, 112.5, 157.5, 22.5, 67.5, 112.5, 157.5]:
                colors.append(j)

        fig = plt.scatter(x=np.transpose(reduced_data)[1],
                          y=np.transpose(reduced_data)[0],
                          c=colors,
                          s=100,
                          cmap="hsv",
                          vmin=0,vmax=180
                          )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(reduction_method.__name__)
        plt.legend(*fig.legend_elements(),
                   loc="lower right", title="Classes")
        plt.show()

    @staticmethod
    def f_pca(x: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=2)
        return pca.fit_transform(x)

    @staticmethod
    def f_tsne(x: np.ndarray) -> np.ndarray:
        tsne = TSNE(n_components=2)
        return tsne.fit_transform(x)

    @staticmethod
    def f_lle(x: np.ndarray) -> np.ndarray:
        lle = LLE(n_components=2, n_neighbors=6)
        return lle.fit_transform(x)

    @staticmethod
    def f_mod_lle(x: np.ndarray) -> np.ndarray:
        lle = LLE(n_components=2, method="modified", n_neighbors=6)
        return lle.fit_transform(x)

    @staticmethod
    def f_hessian_lle(x: np.ndarray) -> np.ndarray:
        lle = LLE(n_components=2, method="hessian", n_neighbors=6)
        return lle.fit_transform(x)

    @staticmethod
    def f_tangent_lle(x: np.ndarray) -> np.ndarray:
        lle = LLE(n_components=2, method="ltsa", n_neighbors=6)
        return lle.fit_transform(x)

    @staticmethod
    def f_isomap(x: np.ndarray) -> np.ndarray:
        iso = Isomap(n_components=2)
        return iso.fit_transform(x)

    @staticmethod
    def f_spectral(x: np.ndarray) -> np.ndarray:
        spec = SpectralEmbedding(n_components=2)
        return spec.fit_transform(x)

    @staticmethod
    def f_mds(x: np.ndarray) -> np.ndarray:
        mds = MDS(n_components=2)
        return mds.fit_transform(x)

    @staticmethod
    def f_umap(x: np.ndarray) -> np.ndarray:
        from umap import UMAP
        umap_2d = UMAP(n_components=2, init='random', random_state=0)
        return umap_2d.fit_transform(x)

    @staticmethod
    def f_cebra(x: np.ndarray) -> np.ndarray:
        cebra_model = CEBRA(
            model_architecture="offset10-model",
            batch_size=16,
            temperature_mode="auto",
            learning_rate=0.001,
            max_iterations=10,
            time_offsets=10,
            output_dimension=2,
            device="cuda_if_available",
            verbose=True
        )
        return cebra_model.fit_transform(x)

    @staticmethod
    def f_rastermap(x: np.ndarray) -> np.ndarray:
        from rastermap import Rastermap, utils
        model = Rastermap(n_PCs=2, n_clusters=100,
                          locality=0.75, time_lag_window=5).fit(x)
        return model.Usv

    def plot_frame(self, frame: np.ndarray):
        sns.heatmap(frame, vmax=.3, square=True, cmap="YlGnBu")
        plt.show()


"""roi 0 np.isfinit(frame)
frame = np.empty_like(roi)
frame[roi] 0"""

if __name__ == "__main__":
    ds = Dataset()
    # ds.verify_dataset()
    """ds.task_1(Dataset.f_pca, time_point=2)
    ds.task_1(Dataset.f_tsne, time_point=2)
    ds.task_1(Dataset.f_umap, time_point=2)
    ds.task_1(Dataset.f_lle, time_point=2)
    ds.task_1(Dataset.f_tangent_lle, time_point=2)
    ds.task_1(Dataset.f_hessian_lle, time_point=2)
    ds.task_1(Dataset.f_isomap, time_point=2)
    ds.task_1(Dataset.f_spectral, time_point=2)
    ds.task_1(Dataset.f_mds, time_point=2)
    ds.task_1(Dataset.f_cebra, time_point=2)"""
    ds.task_1(Dataset.f_rastermap, time_point=2)




