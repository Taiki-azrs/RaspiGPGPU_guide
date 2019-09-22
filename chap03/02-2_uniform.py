import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def kernel(asm):
    setup_vpm_write()
    mov(vpm, uniform)
    mov(vpm, uniform)
    mov(vpm, uniform)

    # [VPM->メモリ]:16要素*3行分書き込む
    setup_dma_store(nrows = 3)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    out = drv.alloc((3,16), 'uint32')
    out[:] = 0.0

    print(' out '.center(80, '='))
    print(out)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[1, 2, 3, out.address]
    )

    print(' out '.center(80, '='))
    print(out)