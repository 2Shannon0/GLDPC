import pickle
from trellis4decoder import Trellis


FILE_PATH = "/home/i17m5/GLDPC/matricies/H_ham(16,11).csv"
TRELLIS_NAME = "H_ham(16,11)"


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
