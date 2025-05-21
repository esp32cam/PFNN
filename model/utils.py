# utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import directed_hausdorff
import seaborn as sns


def estimate_lyapunov(x, method="auto"):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(np.unique(np.round(x, 6))) < 10:
        print("⚠️ Degenerate series — may not be chaotic.")

    if method in ["auto", "nolds"]:
        try:
            import nolds
            return nolds.lyap_r(x, emb_dim=3, lag=1)
        except Exception as e:
            print("nolds.lyap_r failed:", e)

    # fallback
    print("⚠️ Using simple fallback method")
    eps = 1e-6
    dx = np.abs(np.diff(x))
    dx[dx < eps] = eps
    return np.mean(np.log(dx / eps))


def multi_embed(data, delay=1, dimension=3):
    N, D = data.shape
    M = N - (dimension - 1) * delay
    embedded = np.zeros((M, D * dimension))
    for i in range(dimension):
        embedded[:, i * D:(i + 1) * D] = data[i * delay: i * delay + M]
    return embedded


def compare_rollout_to_true(z_pred, z_true, label_pred="PFNN", label_true="Real", mode="pca"):
    if mode == "pca":
        pca = PCA(n_components=2)
        all_data = np.vstack([z_pred, z_true])
        z_pca = pca.fit_transform(all_data)
        z_pred_2d = z_pca[:len(z_pred)]
        z_true_2d = z_pca[len(z_pred):]
    else:
        z_pred_2d = z_pred
        z_true_2d = z_true

    # Plot
    plt.figure(figsize=(7, 6))
    plt.plot(z_true_2d[:, 0], z_true_2d[:, 1], alpha=0.6, label=label_true, color='blue')
    plt.plot(z_pred_2d[:, 0], z_pred_2d[:, 1], alpha=0.6, label=label_pred, color='red')
    plt.title("Latent Attractor: Real vs PFNN")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Distance metrics
    mse = mean_squared_error(z_true_2d[:len(z_pred_2d)], z_pred_2d)
    hausdorff_fwd = directed_hausdorff(z_pred_2d, z_true_2d)[0]
    hausdorff_bwd = directed_hausdorff(z_true_2d, z_pred_2d)[0]
    hausdorff = max(hausdorff_fwd, hausdorff_bwd)

    print(f"🔍 MSE (rollout vs true): {mse:.5f}")
    print(f"📐 Hausdorff Distance: {hausdorff:.5f}")


def plot_spectrum(operator, name="Gm"):
    W = operator.weight.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(W)
    plt.figure(figsize=(5, 5))
    plt.scatter(eigvals.real, eigvals.imag, color='red', label='Eigenvalues')
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(f"Spectrum of {name}")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_invariant_density(z_pred, model_name="model"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import gaussian_kde
    plt.figure(figsize=(8, 6))
    try:
        sns.kdeplot(x=z_pred[:, 0], y=z_pred[:, 1], fill=True, cmap="viridis", thresh=0.05)
    except Exception as e:
        print(f"Seaborn kdeplot failed: {e}. Using fallback scatter plot.")
        plt.scatter(z_pred[:, 0], z_pred[:, 1], alpha=0.4, label="Latent")
    plt.title(f"Invariant Density Map ({model_name})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()
