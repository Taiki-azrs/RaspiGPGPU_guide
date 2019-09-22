import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


def astype_int24(array):
    array = np.left_shift(array, 8)
    array = np.right_shift(array, 8)
    return array


@qpu
def kernel(asm):
    # [メモリ->VPM]:16要素*2行分読み込む
    setup_dma_load(nrows = 2)
    start_dma_load(uniform)
    wait_dma_load()

    # [VPM->レジスタ]:16要素*2行分読み込む
    setup_vpm_read(nrows = 2)
    mov(r0, vpm)
    mov(r1, vpm)

    # 演算:r0とr1をかけてvpmに書き込む
    setup_vpm_write()
    imul24(vpm, r0, r1)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    # ランダムな16個の整数で埋めた配列を作る
    list_a = np.random.randint(-2500, 2500, 16).astype('int32')
    list_b = np.random.randint(-2500, 2500, 16).astype('int32')
    
    # 配列の結合
    inp = drv.copy(np.r_[list_a, list_b])
    out = drv.alloc(16, 'int32')

    print(' list_a '.center(80, '='))
    print(list_a)
    print(' list_b '.center(80, '='))
    print(list_b)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[inp.address, out.address]
    )

    # int24表現にキャスト
    out = astype_int24(out)

    print(' out '.center(80, '='))
    print(out)

    cpu_ans = list_a * list_b
    error   = cpu_ans - out
    print(' error '.center(80, '='))
    print(np.abs(error))