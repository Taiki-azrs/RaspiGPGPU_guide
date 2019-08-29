# coding:utf-8
#TODO 転送できていない原因
import numpy as np
#np.set_printoptions(threshold=np.inf)
import time
from videocore.assembler import qpu
from videocore.driver import Driver
def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values
def thr_mask(idx1,idx2,idx3):
    values = [1]*16
    values[idx1] = 0
    values[idx2] = 0
    values[idx3] = 0
    return values
@qpu
def pimatrix(asm):
    A_ADDR=0
    B_ADDR=1
    C_ADDR=2
    STR=3
    SIMD_ITER_H=4
    IO_ITER_W=5
    THR_ID=6
    THR_NM=7
    W_BACKUP=8
    COMPLETED=0

    mov(r2,1)
    ldi(null,mask(A_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(B_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(C_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(STR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(SIMD_ITER_H),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(IO_ITER_W),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_ID),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')
    rotate(broadcast,r2,-IO_ITER_W)
    ldi(null,mask(W_BACKUP),set_flags=True)
    mov(r2,r5,cond='zs')


    
    rotate(broadcast,r2,-STR)
    imul24(r3,element_number,r5)
    rotate(broadcast,r2,-A_ADDR)
    iadd(r0,r5,r3)
    rotate(broadcast,r2,-B_ADDR)
    iadd(r1,r5,r3)
    #r0:A_ADDR
    #r1:B_ADDR

    L.Hloop    
    L.Wloop
    
    for i in range(32):
        #ra
        mov(tmu0_s,r0)
        mov(tmu1_s,r1)
        nop(sig='load tmu0')
        iadd(r0,r0,4)
        iadd(r1,r1,4)
        mov(r3,r4,sig='load tmu1')
        fadd(ra[i],r3,r4)


        #rb
        mov(tmu0_s,r0)
        mov(tmu1_s,r1)
        nop(sig='load tmu0')
        iadd(r0,r0,4)
        iadd(r1,r1,4)
        mov(r3,r4,sig='load tmu1')
        fadd(rb[i],r3,r4)



    #転送幅の決定
    rotate(broadcast,r2,-STR)
    ldi(r3,16*4)
    isub(broadcast,r5,r3)

    mutex_acquire()
    setup_dma_store_stride(r5,tmp_reg=r3)
    rotate(broadcast,r2,-C_ADDR)
    setup_vpm_write(mode='32bit vertical',Y=0,X=0)
    ldi(r3,16*4)
    for i in range(8):
        mov(vpm,ra[i])
        mov(vpm,rb[i])
    setup_dma_store(mode='32bit horizontal',Y=0,nrows=16)
    start_dma_store(r5)
    iadd(broadcast,r5,r3)
    
    for i in range(1,4):
        setup_vpm_write(mode='32bit vertical',Y=16*i,X=0)
        for j in range(8*i,8*(i+1)):
            mov(vpm,ra[j])
            mov(vpm,rb[j])
        wait_dma_store()
        setup_dma_store(mode='32bit horizontal',Y=16*i,nrows=16)
        mov(vpm_st_addr,r5)
        iadd(broadcast,r5,r3)
    mutex_release()




    ldi(null,mask(IO_ITER_W),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.Wloop)
    ldi(null,mask(C_ADDR),set_flags=True)
    mov(r2,r5,cond='zs')
    nop()

    
    rotate(broadcast,r2,-STR)
    ldi(r3,15)
    imul24(r3,r3,r5)
    ldi(null,mask(C_ADDR),set_flags=True)
    iadd(r2,r2,r3,cond='zs')
    
    rotate(broadcast,r2,-W_BACKUP)
    ldi(null,mask(IO_ITER_W),set_flags=True)
    mov(r2,r5,cond='zs')

    
    ldi(null,mask(SIMD_ITER_H),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.Hloop)
    iadd(r0,r0,r3)
    iadd(r1,r1,r3)
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
    th_H=int(H/n_threads)
    simd_iter_H=int(th_H/SIMD)
    io_iter_W=int(W/64)
    A=drv.alloc((H,W),'float32')
    B=drv.alloc((H,W),'float32')
    C=drv.alloc((H,W),'float32')
    C[:]=0.0
    A[:]=np.random.randn(H,W)
    B[:]=np.random.randn(H,W)
    start = time.time()
    CC=A+B
    elapsed_cpu = time.time() - start
    uniforms=drv.alloc((n_threads,12),'uint32')
    for th in range(n_threads):
        uniforms[th,0]=A.addresses()[int(th_H*th),0]
        uniforms[th,1]=B.addresses()[int(th_H*th),0]
        uniforms[th,2]=C.addresses()[int(th_H*th),0]
    uniforms[:,3]=A.strides[0]
    uniforms[:,4]=simd_iter_H
    uniforms[:,5]=io_iter_W
    uniforms[:,6]=np.arange(1,(n_threads+1))
    uniforms[:,7]=n_threads
    code=drv.program(pimatrix)
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
