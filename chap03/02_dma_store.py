import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def dma_store(asm):
    # [レジスタ->VPM]
    setup_vpm_write() # == setup_vpm_write(X=0, Y=0, mode='32bit horizontal')
    mov(vpm, 1.0)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1) # == setup_dma_store(X=0, Y=0, nrows=1, ncols=16, mode='32bit horizontal')
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    out = drv.alloc(16, 'float32')
    out[:] = 0.0

    print(' out_Before '.center(80, '='))
    print(out)

    drv.execute(
        n_threads=1,
        program  =drv.program(dma_store),
        uniforms =[out.address]
    )

    print(' out_After '.center(80, '='))
    print(out)