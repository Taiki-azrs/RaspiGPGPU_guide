import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values


@qpu
def kernel(asm):
    A_ADDR   =0 #インデックス
    OUT_ADDR =1

    #r2にuniformを格納
    ldi(null,mask(A_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(OUT_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')

    nop()  # rotate命令の直前に回転するアキュムレータ(今回はr2)に値を書き込んではいけない
    
    rotate(broadcast,r2,-A_ADDR)
    # r5=[list_a.address, list_a.address, ..... list_a.address]

    imul24(r3,element_number,4)
    # r3=[0,4,8 ... 52,56,60]

    iadd(r0,r5,r3)
    # この計算により、4byteエレメントの連番アドレスが得られる

    setup_vpm_write()
    mov(vpm, r0)

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    rotate(broadcast,r2,-OUT_ADDR)
    start_dma_store(r5)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc(16, 'float32')
    list_a[:] = 0.0

    out = drv.alloc(16, 'uint32')

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    print("list_a.address:", list_a.address)

    print(' out '.center(80, '='))
    for i,addr in enumerate(out):
      print(f"{i:02d}: {addr}")