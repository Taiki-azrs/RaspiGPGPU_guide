import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def kernel(asm):
    # VPM使うことは確定なので最初にセットアップしておく
    setup_vpm_write()

    # [メモリ->VPM]:16要素*2行分読み込む
    setup_dma_load(nrows = 2)
    start_dma_load(uniform)
    wait_dma_load()

    # [VPM->レジスタ]:16要素*2行分読み込む
    setup_vpm_read(nrows = 2)
    mov(ra0, vpm)
    mov(rb0, vpm)
    nop() # rb0が書き込んですぐ読めないため1命令待つ

    fadd(r2, ra0, rb0).fmul(r3, ra0, rb0)

    mov(vpm, r2)
    mov(vpm, r3)

    # [VPM->メモリ]:16要素*2行分書き込む
    setup_dma_store(nrows = 2)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc((2,16), 'float32')
    list_a[0][:] = 3.0
    list_a[1][:] = 4.0

    out = drv.alloc((2,16), 'float32')

    print(' list_a '.center(80, '='))
    print(list_a)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    print(' out '.center(80, '='))
    print(out)