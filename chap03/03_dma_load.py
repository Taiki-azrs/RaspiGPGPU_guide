import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def kernel(asm):
    # [メモリ->VPM]:16要素*1行分読み込む
    setup_dma_load(nrows = 1) # == setup_dma_load(X=0, Y=0, mode='32bit horizontal')
    start_dma_load(uniform)
    wait_dma_load()

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc(16, 'float32')
    list_a[:] = np.arange(1, 17)

    out = drv.alloc(16, 'float32')
    out[:] = 0.0

    print(' list_a '.center(80, '='))
    print(list_a)
    print(' out_Before '.center(80, '='))
    print(out)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    print(' out_After '.center(80, '='))
    print(out)