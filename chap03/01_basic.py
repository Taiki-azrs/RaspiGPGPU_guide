import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver

@qpu
def basic_temp(asm):
    # ここにカーネルプログラムを書く

    nop() # なにもしない

    # GPUコードを終了する
    exit()


with Driver() as drv:
    # ここにホストプログラムを書く

    inp = drv.alloc(16, 'float32')  # GPUメモリの確保
    inp[:] = 0.0

    # カーネルプログラム実行
    drv.execute(
            n_threads=1,   # QPUのスレッド数(1~12)，
            program  =drv.program(basic_temp), # 実行するカーネルプログラム
            uniforms =[inp.address] # uniformにセットする初期値
            )