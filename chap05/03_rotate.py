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
    # r0 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    # rotate命令の直前に回転するアキュムレータ(今回はr0)に値を書き込んではいけない
    nop()
    
    rotate(r1, r0, 2)
    # r1 = [15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    mov(vpm, r1)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc(16, 'float32')
    list_a[:] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

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