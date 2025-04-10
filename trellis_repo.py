import pickle
from trellis4decoder import Trellis


FILE_PATH = "/home/i17m5/GLDPC/matricies/BCH_MATRIX_N_15_K_11_DEFAULT.csv"
TRELLIS_NAME = "BCH_MATRIX_N_15_K_11_DEFAULT"


def save_trellis(file_path, trellis_name):
    t_h = Trellis(matrix_file=file_path)
    t_h.build_trellis()
    # t_h.plot_sections()

    with(open(f'trellis_binaries/{trellis_name}', "wb")) as f:
        pickle.dump(t_h, f)
    print(f'\nРешетка "{trellis_name}" сохранена!')


def get_trellis(trellis_name):
    with(open(trellis_name, "rb")) as f:
        trellis = pickle.load(f)
        return trellis

# save_trellis(FILE_PATH, TRELLIS_NAME)
