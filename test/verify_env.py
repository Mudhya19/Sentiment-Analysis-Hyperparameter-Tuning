import tensorflow as tf
import optuna
import sklearn
import pandas
import seaborn

print("=== VERIFIKASI ENVIRONMENT .venv ===")
print(f"  TensorFlow  : {tf.__version__}")
print(f"  Optuna      : {optuna.__version__}")
print(f"  Scikit-learn: {sklearn.__version__}")
print(f"  Pandas      : {pandas.__version__}")
print(f"  Seaborn     : {seaborn.__version__}")
gpus = tf.config.list_physical_devices("GPU")
print(f"  GPU         : {gpus if gpus else 'Tidak ada (CPU mode)'}")
print("=== SEMUA LIBRARY SIAP DIGUNAKAN ===")
