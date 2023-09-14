import os, sys
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Any, Tuple
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, MDS
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from cebra import CEBRA
import matplotlib.pyplot as plt
import colorcet as cc
from tqdm import tqdm

BP = os.path.realpath(os.path.join(os.path.realpath(__file__), "../.."))


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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
            print(20 * "=")
            print(f"{day}:")
            print(f"binocular-shape: {self.data[day].binocular.shape}")
            print(f"contra-shape: {self.data[day].contra.shape}")
            print(f"ipsi-shape: {self.data[day].ipsi.shape}")
            print(f"spont-shape: {self.data[day].spont.shape}")
            print(20 * "=")

    def dim_reduction(self,
                      reduction_method: Callable[[np.ndarray, int, Optional[Any]], np.ndarray],
                      day: str = "Day_EO+6",
                      time_point: int = 2,
                      dim: int = 2,
                      save: bool = True,
                      do_train: bool = False,
                      classifier: Optional[Callable[[np.ndarray, np.ndarray], Tuple[float, float]]] = None,
                      data_type: str = "bin",
                      do_plot: bool = True,
                      show_plot: bool = True
                      ) -> Optional[float]:
        if data_type == "bin":
            # zB (8, 18, 8, 135, 160)
            orig_shape = self.data[day].binocular.shape
            # print(orig_shape)
            d1 = self.data[day].binocular
        elif data_type == "contra":
            orig_shape = self.data[day].contra.shape
            d1 = self.data[day].contra
        elif data_type == "ipsi":
            orig_shape = self.data[day].ipsi.shape
            d1 = self.data[day].ipsi
        else:
            raise Exception("Specify valid dataset")
        # preprocess
        d1 = np.reshape(d1, d1.shape[:3] + tuple([d1.shape[-1] * d1.shape[-2]]))
        d1 = np.reshape(d1[time_point], (d1[time_point].shape[0] * d1[time_point].shape[1], d1[time_point].shape[-1]))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', keep_empty_features=True, fill_value=0)
        d1 = imp_mean.fit_transform(d1)
        x = preprocessing.normalize(d1)

        colors = []
        """cc = 3*[0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
        for i in range(orig_shape[1]):
            colors.extend(8*[cc[i]])
                
        for i in range(orig_shape[1]):
            for j in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
                colors.append(j)
        """  # used
        for i in range(orig_shape[1] // 2):

            # for j in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
            for j in [0, 45, 90, 135, 0, 45, 90, 135]:
                # for j in [0, 22.5, 45, 67.6, 0, 22.5, 45, 67.6]:
                colors.append(j)
        for i in range(orig_shape[1] // 2, orig_shape[1]):

            # for j in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]:
            for j in [22.5, 67.5, 112.5, 157.5, 22.5, 67.5, 112.5, 157.5]:
                # for j in [90, 112.5, 135, 157.5, 90, 112.5, 135, 157.5]:
                colors.append(j)

        colors = np.array(colors)
        # Dim-Reduction:
        if reduction_method.__name__ == "f_cebra":
            reduced_data = reduction_method(x, dim, labels=colors)
        else:
            reduced_data = reduction_method(x, dim)
        reduced_data = preprocessing.normalize(reduced_data)
        f1 = None
        if do_train:
            acc, f1 = classifier(reduced_data, colors)
        if do_plot:
            # plot for 2 dimensoins
            if dim == 2:
                fig = plt.scatter(x=np.transpose(reduced_data)[1],
                                  y=np.transpose(reduced_data)[0],
                                  c=colors,
                                  s=100,
                                  cmap="hsv",
                                  vmin=0,
                                  vmax=180
                                  )
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title(f"{reduction_method.__name__.split('_')[-1]}_{day}_dim={dim}_tp={time_point}")
                plt.legend(*fig.legend_elements(),
                           title="Classes", bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.tight_layout()
                if save:
                    os.makedirs(os.path.join(BP, "data", f"{day}__tp={time_point}"), exist_ok=True)
                    plt.savefig(os.path.join(BP, "data", f"{day}__tp={time_point}",
                                             f"{reduction_method.__name__.split('_')[-1]}_dim={dim}.png"))
                if show_plot:
                    plt.show()
                else:
                    plt.close("all")
            elif dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                sc = ax.scatter(*np.transpose(reduced_data),
                                c=colors,
                                s=100,
                                cmap="hsv",
                                vmin=0,
                                vmax=180
                                )
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.ylabel("Z")
                plt.title(f"{reduction_method.__name__.split('_')[-1]}_{day}_dim={dim}_tp={time_point}")
                plt.legend(*sc.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.tight_layout()
                if save:
                    os.makedirs(os.path.join(BP, "data", f"{day}__tp={time_point}"), exist_ok=True)
                    plt.savefig(os.path.join(BP, "data", f"{day}__tp={time_point}",
                                             f"{reduction_method.__name__.split('_')[-1]}_dim={dim}.png"))
                if show_plot:
                    plt.show()
                else:
                    plt.close("all")
        return f1

    @staticmethod
    def train_svm(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        from sklearn import svm
        from sklearn.metrics import classification_report
        from sklearn.utils import shuffle
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        x, y = shuffle(x, y, random_state=0)
        x_train = x[:int(len(x) * 0.8)]
        x_test = x[int(len(x) * 0.8):]
        y_train = y[:int(len(y) * 0.8)]
        y_test = y[int(len(y) * 0.8):]
        # train
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(x_train, y_train)
        # test
        y_pred = clf.predict(x_test)
        res = classification_report(y_test, y_pred, output_dict=True)
        return res["accuracy"], res["weighted avg"]["f1-score"]

    @staticmethod
    def train_knn(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import classification_report
        from sklearn.utils import shuffle
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        x, y = shuffle(x, y, random_state=0)
        x_train = x[:int(len(x) * 0.8)]
        x_test = x[int(len(x) * 0.8):]
        y_train = y[:int(len(y) * 0.8)]
        y_test = y[int(len(y) * 0.8):]
        # train
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(x_train, y_train)
        # test
        y_pred = knn.predict(x_test)
        res = classification_report(y_test, y_pred, output_dict=True)
        return res["accuracy"], res["weighted avg"]["f1-score"]

    @staticmethod
    def do_nothing(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        return x

    @staticmethod
    def f_pca(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        pca = PCA(n_components=dim)
        return pca.fit_transform(x)

    @staticmethod
    def pca_explain(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        pca = PCA(n_components=dim)
        pca_data = pca.fit_transform(x)
        print("compo")
        print(pca.components_)
        print()
        print(pca.explained_variance_)
        return pca_data

    @staticmethod
    def f_tsne(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        tsne = TSNE(n_components=dim, perplexity=70)
        return tsne.fit_transform(x)

    @staticmethod
    def f_lle(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        lle = LLE(n_components=dim, n_neighbors=int(dim * (dim + 3) / 2) + 1)
        return lle.fit_transform(x)

    @staticmethod
    def f_mod_lle(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        lle = LLE(n_components=dim, method="modified", n_neighbors=int(dim * (dim + 3) / 2) + 1)
        return lle.fit_transform(x)

    @staticmethod
    def f_hessian_lle(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        lle = LLE(n_components=dim, method="hessian", n_neighbors=int(dim * (dim + 3) / 2) + 1)
        return lle.fit_transform(x)

    @staticmethod
    def f_tangent_lle(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        lle = LLE(n_components=dim, method="ltsa", n_neighbors=int(dim * (dim + 3) / 2) + 1)
        return lle.fit_transform(x)

    @staticmethod
    def f_isomap(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        iso = Isomap(n_components=dim)
        return iso.fit_transform(x)

    @staticmethod
    def f_spectral(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        spec = SpectralEmbedding(n_components=dim)
        return spec.fit_transform(x)

    @staticmethod
    def f_mds(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        mds = MDS(n_components=dim)
        return mds.fit_transform(x)

    @staticmethod
    def f_umap(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        from umap import UMAP
        umap_2d = UMAP(n_components=dim, init='random', random_state=0)
        return umap_2d.fit_transform(x)

    @staticmethod
    def f_cebra(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        cebra_model = CEBRA(
            model_architecture="offset10-model",
            batch_size=16,
            temperature_mode="auto",
            learning_rate=0.001,
            max_iterations=10,
            time_offsets=10,
            output_dimension=dim,
            device="cuda_if_available",
            verbose=True
        )
        return cebra_model.fit_transform(x, kwargs["labels"])

    @staticmethod
    def f_rastermap(x: np.ndarray, dim: int, **kwargs) -> np.ndarray:
        from rastermap import Rastermap, utils
        model = Rastermap(n_PCs=dim, n_clusters=100,
                          locality=0.75, time_lag_window=5).fit(x)
        return model.Usv

    def plot_frame(self, frame: np.ndarray):
        sns.heatmap(frame, vmax=.3, square=True, cmap="YlGnBu")
        plt.show()

    @staticmethod
    def animate_plots(day: str,
                      function_name: str = "pca",
                      dim: int = 2,
                      interval=6000):
        from matplotlib.animation import FuncAnimation
        nframes = 8
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
        os.makedirs(os.path.join(BP, "data", "gifs", day), exist_ok=True)

        def animate(i):
            im = plt.imread(os.path.join(BP, "data", f"{day}__tp={i}", f"{function_name}_dim={dim}.png"))
            plt.imshow(im)

        anim = FuncAnimation(plt.gcf(), animate, frames=nframes,
                             interval=(interval / nframes))
        anim.save(filename=os.path.join(BP, f"data/gifs/{day}", f"{day}_{function_name}_dim={dim}.gif"),
                  writer='imagemagick')

    def run_all(self,
                day: str = "Day_EO+6",
                time_point: int = 2,
                save: bool = True,
                dt: str = "bin"
                ):
        for method in [i for i in dir(self) if i[:2] == "f_" and callable(getattr(self, i))]:
            method = getattr(self, method)
            self.dim_reduction(reduction_method=method, day=day, time_point=time_point, dim=2, save=save, data_type=dt,
                               show_plot=False)

        for method in [i for i in dir(self) if i[:2] == "f_" and callable(getattr(self, i))]:
            method = getattr(self, method)
            self.dim_reduction(reduction_method=method, day=day, time_point=time_point, dim=3, save=save, data_type=dt,
                               show_plot=False)

    def run_all_whole_day(self, day: str, dt: str = "bin"):
        for t in range(8):
            ds.run_all(day=day, time_point=t, dt=dt)
        for method in [i for i in dir(self) if i[:2] == "f_" and callable(getattr(self, i))]:
            method = method.split("_")[-1]
            self.animate_plots(day=day, dim=2, function_name=method)
            self.animate_plots(day=day, dim=3, function_name=method)

    def run(self):
        import warnings
        warnings.filterwarnings('ignore')
        x = ["-2", "+0", "+2", "+4", "+6"]
        """bar = tqdm(total=3*len(x))
        for dt in ["bin", "contra", "ipsi"]:
            for i in x:
                with HiddenPrints():
                    self.run_all_whole_day(f"Day_EO{i}", dt=dt)
                bar.update(1)"""
        bar = tqdm(total=len(x))
        for i in x:
            with HiddenPrints():
                self.run_all_whole_day(f"Day_EO{i}", dt="bin")
            bar.update(1)

    def train_day_fixed(self,
                        classifier: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
                        day: str, dim=2,
                        ):
        plot_colors = sns.color_palette(cc.glasbey_light, n_colors=20)
        x = [i for i in range(8)]
        for idx, method in enumerate(
                [i for i in dir(self) if i[:2] == "f_" and callable(getattr(self, i))] + ["do_nothing"]):
            if method in ["f_rastermap"]:
                pass
            else:
                y = []
                method = getattr(self, method)
                with HiddenPrints():
                    for j in x:
                        if method.__name__ == "do_nothing":
                            y.append(self.dim_reduction(method, day, time_point=j, dim=200, save=False, do_train=True,
                                                        do_plot=False, classifier=classifier))
                        else:
                            y.append(self.dim_reduction(method, day, time_point=j, dim=dim, save=False, do_train=True,
                                                        do_plot=False, classifier=classifier))
                plt.plot(x, y, label=method.__name__.split("_")[-1], color=plot_colors[idx % len(plot_colors)])
        plt.xlabel("timepoint")
        plt.ylabel("f1-Score")
        plt.title(f"Day:{day}__Dim:{dim}")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(BP, "data", "classifier_results",
                                 f"{classifier.__name__.split('_')[-1]}_day={day}_dim={dim}.png"))
        plt.show()

    def train_tp_fixed(self,
                       classifier: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
                       tp: int, dim=2):
        plot_colors = sns.color_palette(cc.glasbey_light, n_colors=20)
        x = ["-2", "+0", "+2", "+4", "+6"]
        for idx, method in enumerate(
                [i for i in dir(self) if i[:2] == "f_" and callable(getattr(self, i))] + ["do_nothing"]):
            if method in ["f_rastermap"]:
                pass
            else:
                y = []
                method = getattr(self, method)
                with HiddenPrints():
                    for j in x:
                        if method.__name__ == "do_nothing":
                            y.append(self.dim_reduction(method, day=f"Day_EO{j}", time_point=tp, dim=200, save=False,
                                                        do_train=True,
                                                        do_plot=False, classifier=classifier))
                        else:
                            y.append(self.dim_reduction(method, day=f"Day_EO{j}", time_point=tp, dim=dim, save=False,
                                                        do_train=True,
                                                        do_plot=False, classifier=classifier))
                plt.plot(x, y, label=method.__name__.split("_")[-1], color=plot_colors[idx % len(plot_colors)])
        plt.xlabel("timepoint")
        plt.ylabel("f1-Score")
        plt.title(f"TP:{tp}__Dim:{dim}")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(BP, "data", "classifier_results",
                                 f"{classifier.__name__.split('_')[-1]}_tp={tp}_dim={dim}.png"))
        plt.show()


if __name__ == "__main__":
    ds = Dataset()
    # ds.verify_dataset()
    """
    ds.task_1(Dataset.f_tsne, time_point=2)
    ds.task_1(Dataset.f_umap, time_point=2)
    ds.task_1(Dataset.f_lle, time_point=2)
    ds.task_1(Dataset.f_tangent_lle, time_point=2)
    ds.task_1(Dataset.f_hessian_lle, time_point=2)
    ds.task_1(Dataset.f_isomap, time_point=2)
    ds.task_1(Dataset.f_spectral, time_point=2)
    ds.task_1(Dataset.f_mds, time_point=2)
    ds.task_1(Dataset.f_cebra, time_point=2)
    ds.task_1(Dataset.f_rastermap, time_point=2)
    """
    ds.train_day_fixed(ds.train_knn, day="Day_EO-2", dim=2)
    ds.train_day_fixed(ds.train_knn, day="Day_EO+0", dim=2)
    ds.train_day_fixed(ds.train_knn, day="Day_EO+2", dim=2)
    ds.train_day_fixed(ds.train_knn, day="Day_EO+4", dim=2)
    ds.train_day_fixed(ds.train_knn, day="Day_EO+6", dim=2)
    # ds.run()
    # ds.train_tp_fixed(ds.train_knn, 2, dim=2)
    # ds.train_day_fixed(ds.train_knn, day="Day_EO+4", dim=3)
    # ds.train_tp_fixed(ds.train_knn, tp=6, dim=3)
    # ds.dim_reduction(ds.f_lle, day="Day_EO+6", time_point=2, dim=2)
    # ds.dim_reduction(Dataset.f_cebra, time_point=1, dim=3)
    # ds.dim_reduction(Dataset.f_pca, time_point=2, dim=3)
    # ds.dim_reduction(Dataset.f_cebra, time_point=2, dim=2)
    # ds.dim_reduction(Dataset.f_cebra, time_point=2, dim=3)
    """for t in range(8):
        ds.run_all(day="Day_EO+6", time_point=t)
    f = "isomap"
    ds.animate_plots(day="Day_EO+6", dim=2, function_name=f)
    ds.animate_plots(day="Day_EO+6", dim=3, function_name=f)"""
