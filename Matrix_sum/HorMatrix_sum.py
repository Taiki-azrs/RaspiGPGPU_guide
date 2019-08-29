# coding:utf-8
import numpy as np
import time
from videocore.assembler import qpu
from videocore.driver import Driver
def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values
@qpu
def pimatrix2(asm):
    A_ADDR=0
    B_ADDR=1
    C_ADDR=2
    IO_ITER=3
    THR_ID=4
    THR_NM=5
    COMPLETED=0

    mov(r2,1)
    ldi(null,mask(A_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(B_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(C_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(IO_ITER),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_ID),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')
    
    imul24(r3,element_number,4)
    rotate(broadcast,r2,-A_ADDR)
    iadd(r0,r5,r3)
    rotate(broadcast,r2,-B_ADDR)
    iadd(r1,r5,r3)
    #r0:A_ADDR
    #r1:B_ADDR

    L.loop

    ldi(broadcast,16*4)
    for i in range(32):
        #ra
        mov(tmu0_s,r0)
        mov(tmu1_s,r1)
        nop(sig='load tmu0')
        iadd(r0,r0,r5)
        iadd(r1,r1,r5)
        mov(r3,r4,sig='load tmu1')
        fadd(ra[i],r3,r4)


        #rb
        mov(tmu0_s,r0)
        mov(tmu1_s,r1)
        nop(sig='load tmu0')
        iadd(r0,r0,r5)
        iadd(r1,r1,r5)
        mov(r3,r4,sig='load tmu1')
        fadd(rb[i],r3,r4)



    ldi(r3,64*16*4)

    mutex_acquire()
    rotate(broadcast,r2,-C_ADDR)
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0)
    for i in range(32):
        mov(vpm,ra[i])
        mov(vpm,rb[i])
    setup_dma_store(mode='32bit horizontal',Y=0,nrows=64)
    start_dma_store(r5)
    wait_dma_store()
    mutex_release()


    ldi(null,mask(C_ADDR),set_flags=True)
    iadd(r2,r2,r3,cond='zs')
    ldi(null,mask(IO_ITER),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.loop)
    nop()
    nop()
    nop()



#====semafo=====    
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
    sema_down(COMPLETED)    # Wait completion of all threads.
    nop()
    iadd(r0, r0, -1)
    
    interrupt()
    
    L.skip_fin
    
    exit(interrupt=False)

    
with Driver() as drv:
    H=1920
    W=1088
    n_threads=12
    SIMD=16
    R=64
    th_H=int(H/n_threads)
    th_ele=th_H*W
    io_iter=int(th_ele/(R*SIMD))
    A=drv.alloc((H,W),'float32')
    B=drv.alloc((H,W),'float32')
    C=drv.alloc((H,W),'float32')
    C[:]=0.0
    A[:]=np.random.randn(H,W)
    B[:]=np.random.randn(H,W)
    start = time.time()
    CC=A+B
    elapsed_cpu = time.time() - start
    uniforms=drv.alloc((n_threads,6),'uint32')
    for th in range(n_threads):
        uniforms[th,0]=A.addresses()[int(th_H*th),0]
        uniforms[th,1]=B.addresses()[int(th_H*th),0]
        uniforms[th,2]=C.addresses()[int(th_H*th),0]
    uniforms[:,3]=int(io_iter)
    uniforms[:,4]=np.arange(1,(n_threads+1))
    uniforms[:,5]=n_threads
    code=drv.program(pimatrix2)
    start = time.time()
    drv.execute(
            n_threads=n_threads,
            program=code,
            uniforms=uniforms
            )
    elapsed_gpu = time.time() - start
    print ("GPU:elapsed_time:{0}".format(elapsed_gpu*1000) + "[msec]")
    print ("CPU:elapsed_time:{0}".format(elapsed_cpu*1000) + "[msec]")
    print("{0}Gflops".format((1920*1088)/elapsed_gpu/(1000**3)))
    C=C[:]
    CC=CC[:]
    print(C)
    print(CC)
    print('maximum absolute error: {:.4e}'.format(
        float(np.max(np.abs(C - CC)))))
