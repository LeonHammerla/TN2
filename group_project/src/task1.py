import os
from dataclasses import dataclass
from typing import Optional, Dict
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn . decomposition import PCA

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

    def task_1(self):
        orig_shape = self.data["Day_EO-2"].binocular.shape
        d1 = self.data["Day_EO-2"].binocular
        # preprocess
        d1 = np.reshape(d1, d1.shape[:3]+tuple([d1.shape[-1] * d1.shape[-2]]))
        d1 = zscore(d1, axis=-1, nan_policy="omit")
        # PCA:
        x = np.reshape(d1[2], (d1[2].shape[0]*d1[2].shape[1], d1[2].shape[-1]))
        pca = PCA(n_components=orig_shape[-1] * orig_shape[-2])
        pca_data = pca.fit_transform(x)
        fig = plt.scatter(*np.transpose(pca_data), c=d1[])

    def plot_frame(self, frame: np.ndarray):
        sns.heatmap(frame, vmax=.3, square=True, cmap="YlGnBu")
        plt.show()


"""roi 0 np.isfinit(frame)
frame = np.empty_like(roi)
frame[roi] 0"""

if __name__ == "__main__":
    ds = Dataset()
    ds.task_1()