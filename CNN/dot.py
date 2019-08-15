# coding:utf-8

import numpy as np
import time
import math
from videocore.assembler import qpu
from videocore.driver import Driver

Iter_W=0

def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values
@qpu
def dot(asm): #test
    A_ADDR=0
    B_ADDR=1
    C_ADDR=2
    Q_DEV=3
    P=4
    Q=5
    R=6
    THR=7
    THR_NM=8
    R_STR=9
    UNI_ADDR=10
    
    COMPLETED = 0
    
    #set uniform to r2
    mov(r0,uniform)
    mov(r2,1)
    ldi(null,mask(A_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(B_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(C_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(Q_DEV),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(P),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(Q),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(R),set_flags=True)
    mov(r2,uniform,cond='zs')    
    ldi(null,mask(THR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(R_STR),set_flags=True)
    mov(r2,uniform,cond='zs')

    
    ldi(ra0,16*4)

    
    ldi(null,mask(UNI_ADDR),set_flags=True)
    mov(r2,r0,cond='zs')
    #set uniform end
    
    ldi(ra1,60*4)
    
    rotate(broadcast,r2,-Q_DEV)
    mov(ra2,r5)
    
    rotate(broadcast,r2,-B_ADDR)
    imul24(r1,element_number,4)
    iadd(r1,r1,r5)
    
    ldi(ra4,0.0)


    
    L.H
    mov(uniforms_address,r2)
    ldi(ra3,60)
    
    L.H_uni
    
    mov(tmu0_s,r1)
    nop()
    nop(sig='load tmu0')
    mov(r0,uniform)
    fmul(rb0,r0,r4)
    nop()
    fadd(ra4,rb0,ra4)
    nop()
    """
    for i in range(Iter_W-1):
        iadd(r1,r1,ra0)
        mov(tmu0_s,r1)
        nop()
        nop(sig='load tmu0')
        fmul(rb[i+1],r4,r0)
    """
    rotate(broadcast,r2,-R_STR)
    iadd(r1,r1,r5)
    
    isub(ra3,ra3,1)
    jzc(L.H_uni)
    nop()
    nop()
    nop()
    nop()

    ldi(null,mask(A_ADDR),set_flags=True)
    iadd(r2,r2,ra1,cond='zs')
    isub(ra2,ra2,1)
    jzc(L.H)
    nop()
    nop()
    nop()
    
    mutex_acquire() #排他制御 vpmは一つのため

    setup_vpm_read(mode='32bit horizontal',Y=0,X=0,nrows=Iter_W) #loadしたDMAをvpmにread
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0) #書き込めるようにする
    #for i in range(Iter_W):  #ra,rbをvpmにmov
    #    mov(vpm,rb[i])
    mov(vpm,ra4)
    rotate(broadcast,r2,-C_ADDR)
    setup_dma_store(mode='32bit horizontal',nrows=Iter_W)
    start_dma_store(r5)
    #ldi(null,mask(C_ADDR),set_flags=True)
    #iadd(r2,r5,ra0,cond='zs')
    wait_dma_store()
    
    mutex_release()
        



#====semafo=====    　すべてのスレッドが終わるまで待つ　詳細はquita見て
    sema_up(COMPLETED)
    rotate(broadcast,r2,-THR)
    iadd(null,r5,-1,set_flags=True)
    jzc(L.skip_fin)
    nop()
    nop()
    nop()
    rotate(broadcast,r2,-THR_NM)
    iadd(r1, r5, -1,set_flags=True)
    L.sem_down
    jzc(L.sem_down)
    sema_down(COMPLETED)    # Wait completion of all threads.
    nop()
    iadd(r1, r1, -1)
    
    interrupt()
    
    L.skip_fin
    
    exit(interrupt=False)
def GPU_dot(col,col_W):    
    with Driver() as drv:
        p=1
        q=4320
        r=100
        A=drv.alloc((p,q),'float32')
        B=drv.alloc((q,r),'float32')
        C=drv.alloc((p,r),'float32')
        A[:]=col[:]
        B[:]=col_W[:]
        CC=np.dot(A,B)
        n_threads=10
        r_dev=r/n_threads
        q_dev=int(math.ceil(q/60.0))
        global Iter_W
        Iter_W=int(math.ceil(r_dev/16.0))
        uniforms=drv.alloc((n_threads,11),'uint32')
        uniforms[:,0]=uniforms.addresses()[:,0]
        for th in range(n_threads):
            uniforms[th,1]=A.addresses()[0,0]
            uniforms[th,2]=B.addresses()[0,int(th*r_dev)]
            uniforms[th,3]=C.addresses()[0,int(th*r_dev)]
        uniforms[:,4]=q_dev
        uniforms[:,5]=p
        uniforms[:,6]=q
        uniforms[:,7]=r
        uniforms[:,8]=np.arange(1,(n_threads+1))
        uniforms[:,9]=n_threads
        uniforms[:,10]=B.strides[0]
        code=drv.program(dot)
        elapsed_gpu=0
        start = time.time()
        drv.execute(
            n_threads=n_threads,
            program=code,
            uniforms=uniforms
        )
        elapsed_gpu += time.time() - start
        print ("elapsed_time:{0}".format(elapsed_gpu/100) + "[sec]")
        print(C)
        print(CC)
        print('minimum absolute error: {:.4e}'.format(
            float(np.min(np.abs(CC - C)))))
        print('maximum absolute error: {:.4e}'.format(
            float(np.max(np.abs(CC - C)))))
        CC[:]=C[:]
        return CC
