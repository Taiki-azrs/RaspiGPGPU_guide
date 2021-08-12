import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


def astype_int24(array):
    array = np.left_shift(array, 8)
    array = np.right_shift(array, 8)
    return array


@qpu
def kernel(asm):
    # VPM使うことは確定なので最初にセットアップしておく
    setup_vpm_write()

    # [list_a.address]がr2読み込まれます
    mov(r2, uniform)

    # [VPM->レジスタ]:16要素*1行分読み込む
    setup_vpm_read(nrows = 1)
    mov(r0, vpm)

    imul24(r0, element_number, 4)

    mov(vpm, r0)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc(16, 'int32')
    list_a[:] = 0.0

    out = drv.alloc(16, 'int32')

    print(' list_a '.center(80, '='))
    print(list_a)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    # int24表現にキャスト
    out = astype_int24(out)

    print(' out '.center(80, '='))
    print(out)