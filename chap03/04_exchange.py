import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

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

    # [レジスタ->VPM]:読み込みと逆順に書き込む
    setup_vpm_write()
    mov(vpm, r1)
    mov(vpm, r0)

    # [VPM->メモリ]:16要素*2行分書き込む
    setup_dma_store(nrows = 2)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    # 2行16列の配列を作る
    list_a = drv.alloc((2,16), 'float32')
    list_a[0][:] = 1.0
    list_a[1][:] = 2.0

    out = drv.alloc((2, 16), 'float32')

    print(' list_a '.center(80, '='))
    print(list_a)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    print(' out '.center(80, '='))
    print(out)