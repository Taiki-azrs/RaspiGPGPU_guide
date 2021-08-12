import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


@qpu
def kernel(asm):
    # VPM使うことは確定なので最初にセットアップしておく
    setup_vpm_write()

    # [メモリ->VPM]:16要素*1行分読み込む
    setup_dma_load(nrows = 1)
    start_dma_load(uniform)
    wait_dma_load()

    # [VPM->レジスタ]:16要素*1行分読み込む
    setup_vpm_read(nrows = 1)
    mov(r0, vpm)

    for i in range(5):
      fadd(r0, r0, 1.0)

    mov(vpm, r0)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc(16, 'float32')
    list_a[:] = 0.0

    out = drv.alloc(16, 'float32')

    print(' list_a '.center(80, '='))
    print(list_a)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    print(' out '.center(80, '='))
    print(out)

    for i in range(5):
      list_a += 1.0
    cpu_ans = list_a

    error   = cpu_ans - out
    print(' error '.center(80, '='))
    print(np.abs(error))