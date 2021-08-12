#coding:utf-8
import numpy as np
from PIL import Image, ImageFilter

from videocore.assembler import qpu
from videocore.driver import Driver

def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values

@qpu
def piadd(asm):
    IN_ADDR  =0 #インデックス
    OUT_ADDR =1
    IO_ITER  =2
    THR_ID   =3
    THR_NM   =4
    COMPLETED=0 #セマフォ用

    
    ldi(null,mask(IN_ADDR),set_flags=True)#r2にuniformを格納
    mov(r2,uniform,cond='zs')
    ldi(null,mask(OUT_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(IO_ITER),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_ID),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')
    
    imul24(r3,element_number,4) 
    rotate(broadcast,r2,-IN_ADDR)
    iadd(r0,r5,r3)
    #r0:IN_ADDR

    L.loop

    ldi(broadcast,16*4)
    for i in range(30):
        #ra
        mov(tmu0_s,r0)
        nop(sig='load tmu0')
        iadd(r0,r0,r5)
        mov(ra[i],r4)
        

        #rb
        mov(tmu0_s,r0)
        nop(sig='load tmu0')
        iadd(r0,r0,r5)
        mov(rb[i],r4)


    ldi(r3,60*16*4)

    mutex_acquire()
    rotate(broadcast,r2,-OUT_ADDR)
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0)

    for i in range(30):
        mov(vpm,ra[i])
        mov(vpm,rb[i])

    setup_dma_store(mode='32bit horizontal',Y=0,nrows=60)
    start_dma_store(r5)
    wait_dma_store()

    mutex_release()

    ldi(null,mask(IO_ITER),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.loop)
    ldi(null,mask(OUT_ADDR),set_flags=True)
    iadd(r2,r2,r3,cond='zs')
    nop()



#====semaphore=====    
    sema_up(COMPLETED)
    rotate(broadcast,r2,-THR_ID)
    iadd(null,r5,-1,set_flags=True)
    jzc(L.skip_fin)
    nop()
    nop()
    nop()
    rotate(broadcast,r2,-THR_NM)
    iadd(r0, r5, -1,set_flags=True)
    L.sem_down
    jzc(L.sem_down)
    sema_down(COMPLETED)    # すべてのスレッドが終了するまで待つ
    nop()
    iadd(r0, r0, -1)
    
    interrupt()
    
    L.skip_fin
    
    exit(interrupt=False)

    
with Driver() as drv:
    # 画像サイズ
    H=360
    W=320

    n_threads=12
    SIMD=16
    R=60

    th_H    = int(H/n_threads) #1スレッドの担当行
    th_ele  = th_H*W #1スレッドの担当要素
    io_iter = int(th_ele/(R*SIMD)) #何回転送するか

    IN  = drv.alloc((H,W),'float32')
    OUT = drv.alloc((H,W),'float32')
    OUT[:] = 0.0

    pil_img = Image.open('./LLL.png').convert('L')
    IN[:]   = np.asarray(pil_img)

    CC = IN

    uniforms=drv.alloc((n_threads,5),'uint32')
    for th in range(n_threads):
        uniforms[th,0]=IN.addresses()[int(th_H*th),0]
        uniforms[th,1]=OUT.addresses()[int(th_H*th),0]
    uniforms[:,2]=int(io_iter)
    uniforms[:,3]=np.arange(1,(n_threads+1))
    uniforms[:,4]=n_threads

    code=drv.program(piadd)
    drv.execute(
        n_threads= n_threads,
        program  = drv.program(piadd),
        uniforms = uniforms
    )

    print(OUT-CC)

    pil_img = Image.fromarray(OUT.astype(np.uint8))
    pil_img.save('./out.png')