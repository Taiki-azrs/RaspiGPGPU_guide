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

    # 0x0001とのビット論理積
    band(null, r0, 1)

    # Zフラグがクリアならジャンプ
    jzc(L.odd)
    nop(); nop(); nop()
    # 偶数の場合
    mov(vpm, 2)
    jmp(L.end)
    nop(); nop(); nop()

    # 奇数の場合
    L.odd
    mov(vpm, 1)

    L.end

    # [VPM->メモリ]:16要素*1行分書き込む
    setup_dma_store(nrows = 1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()

with Driver() as drv:
    list_a = drv.alloc(16, 'uint32')
    list_a[:] = 3

    out = drv.alloc(16, 'uint32')

    print(' list_a '.center(80, '='))
    print(list_a)

    drv.execute(
        n_threads=1,
        program  =drv.program(kernel),
        uniforms =[list_a.address, out.address]
    )

    print(' out '.center(80, '='))
    print(out)

    print("Number is ", end="")
    if out[0]==1:
      print("[Odd]")
    elif out[0]==2:
      print("[Even]")
    else:
      print("[error]")