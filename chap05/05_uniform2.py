import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values

@qpu
def kernel(asm):
    # uniformの何番目になんの値があるか
    LA_ADDR = 0
    LB_ADDR = 1
    OUT_ADDR= 2

    ldi(r0, 0.0)

    ldi(null, mask(LA_ADDR), set_flags=True)   #r0にuniformを格納
    mov(r0, uniform, cond='zs')
    ldi(null, mask(LB_ADDR), set_flags=True)   #r0にuniformを格納
    mov(r0, uniform, cond='zs')
    ldi(null, mask(OUT_ADDR), set_flags=True)  #r0にuniformを格納
    mov(r0, uniform, cond='zs')

    # r0=[list_a.address, list_b.address, out.address, 0,0,0,0,0,0,0,0,0,0,0,0,0]


    # [レジスタ->VPM]
    setup_vpm_write()
    mov(vpm, r0)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    rotate(broadcast, r0, -OUT_ADDR)
    start_dma_store(r5)
    wait_dma_store()

    exit()

with Driver() as drv:
    # ランダムな16個の数字で埋めた配列を作る
    list_a = drv.alloc(16, 'float32')
    list_b = drv.alloc(16, 'float32')
    out = drv.alloc(16, 'uint32')

    list_a[:] = 0.0
    list_b[:] = 0.0

    uni = [list_a.address, list_b.address, out.address]
    
    print(' uniforms '.center(80, '='))
    print(f'{uni}')

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =uni
    )
    
    print(' out '.center(80, '='))
    print(out)