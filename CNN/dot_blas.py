# coding:utf-8

import numpy as np
import time
import math
import struct
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
    OUT_ADDR=2
    C_ADDR=3
    Q_DEV=4
    P=5
    Q=6
    R=7
    THR=8
    THR_NM=9
    R_STR=10
    Q_MOD=11
    VPM_FORM=12
    RELU_FLAG=13
    UNI_ADDR=14
    
    COMPLETED = 0
    
    #set uniform to r2
    mov(r0,uniform)
    mov(r2,1)
    ldi(null,mask(A_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(B_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(OUT_ADDR),set_flags=True)
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
    ldi(null,mask(Q_MOD),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(VPM_FORM),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(RELU_FLAG),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(ra0,16*4)

    
    ldi(null,mask(UNI_ADDR),set_flags=True)
    mov(r2,r0,cond='zs')
    #set uniform end
    
    ldi(ra1,64*4)
    
    rotate(broadcast,r2,-Q_DEV)
    mov(ra2,r5)
    
    rotate(broadcast,r2,-B_ADDR)
    imul24(r1,element_number,4)
    iadd(r1,r1,r5)
    
    ldi(ra4,0.0)
    ldi(ra3,64)

    
    L.H
    mov(uniforms_address,r2)

    
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
    ldi(ra3,64)
    nop()
    nop()

    iadd(ra2,ra2,1)
    rotate(broadcast,r2,-Q_MOD)
    ldi(null,mask(Q_MOD),set_flags=True)
    mov(r2,0,cond='zs')
    isub(r3,r5,0)
    jzc(L.H)
    mov(ra3,r5)
    nop()
    nop()

    mutex_acquire() #排他制御 vpmは一つのため
    
    setup_dma_load(mode='32bit horizontal', Y=0, nrows=1, mpitch=0)
    rotate(broadcast,r2,-C_ADDR)
    start_dma_load(r5)

    rotate(broadcast,r2,-VPM_FORM)

    wait_dma_load()

    setup_vpm_read(mode='32bit horizontal',Y=0,X=0,nrows=Iter_W) #loadしたDMAをvpmにread
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0) #書き込めるようにする

    fmul(r0,vpm,r5)
    fadd(r0,ra4,r0)
    rotate(broadcast,r2,-RELU_FLAG)
    mov(ra30,r5)
    jzc(L.relu)
    nop()
    nop()
    nop()
    
    fmax(r0,r0,0) #Relu
    
    L.relu
    
    rotate(broadcast,r2,-OUT_ADDR)
    setup_dma_store(mode='32bit horizontal',nrows=Iter_W)
    start_dma_store(r5)
    #ldi(null,mask(OUT_ADDR),set_flags=True)
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
def GPU_dot(col,col_W,b,Relu_flag=0):
#def main():
    with Driver() as drv:
        p,q=col.shape
        r=col_W.shape[1]
        #p=1
        #q=4320
        #r=100
        A=drv.alloc((p,q),'float32')
        B=drv.alloc((q,r),'float32')
        C=drv.alloc((p,r),'float32')
        out=drv.alloc((p,r),'float32')
        #A[:]=np.random.randn(p,q)
        #B[:]=np.random.randn(q,r)
        #C[:]=np.random.randn(p,r)
        A[:]=col
        B[:]=col_W
        C[:]=b[:]
        CC=np.dot(A,B)+C
        n_threads=12
        r_dev=r/n_threads
        q_dev=int(q/64)
        q_mod=q%64
        vpm_form=1.0
        global Iter_W
        Iter_W=int(math.ceil(r_dev/16.0))
        uniforms=drv.alloc((n_threads,16),'uint32')
        uniforms[:,0]=uniforms.addresses()[:,0]
        for th in range(n_threads):
            uniforms[th,1]=A.addresses()[0,0]
            uniforms[th,2]=B.addresses()[0,int(th*r_dev)]
            uniforms[th,3]=out.addresses()[0,int(th*r_dev)]
            uniforms[th,4]=C.addresses()[0,int(th*r_dev)]
        uniforms[:,5]=q_dev
        uniforms[:,6]=p
        uniforms[:,7]=q
        uniforms[:,8]=r
        uniforms[:,9]=np.arange(1,(n_threads+1))
        uniforms[:,10]=n_threads
        uniforms[:,11]=B.strides[0]
        uniforms[:,12]=int(q_mod)
        uniforms[:,13]=struct.unpack('L',struct.pack('f',vpm_form))[0]
        uniforms[:,14]=Relu_flag
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
        print(out)
        print(CC)
        print('minimum absolute error: {:.4e}'.format(
            float(np.min(np.abs(CC - out)))))
        print('maximum absolute error: {:.4e}'.format(
            float(np.max(np.abs(CC - out)))))
        CC[:]=out[:]
        return CC
