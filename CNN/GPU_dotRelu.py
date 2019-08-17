# coding:utf-8

import numpy as np
import time
import math
import struct
from videocore.assembler import qpu
from videocore.driver import Driver



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
    Q_UNI_ITER=4
    Q_UNI_MOD=5
    R_SIMD_ITER=6
    THR_ID=7
    THR=8
    R_STR=9
    VPM_FORM=10
    RELU_FLAG=11
    B_ADDR_INDEX=12
    
    COMPLETED = 0
    
    #set uniform to r2
    mov(r2,1)
    ldi(null,mask(A_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(B_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(OUT_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(C_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(Q_UNI_ITER),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(Q_UNI_MOD),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(R_SIMD_ITER),set_flags=True)
    mov(r2,uniform,cond='zs')  
    ldi(null,mask(THR_ID),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(R_STR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(VPM_FORM),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(RELU_FLAG),set_flags=True)
    mov(r2,uniform,cond='zs')
    rotate(broadcast,r2,-B_ADDR)
    ldi(null,mask(B_ADDR_INDEX),set_flags=True)
    mov(r2,r5,cond='zs')
    #set uniform end
    
    for i in range(32):
        mov(rb[i],0.0)
    
    #TMU準備
    rotate(broadcast,r2,-B_ADDR)
    imul24(r0,element_number,4)
    iadd(r0,r0,r5)

    #set r3 uniformのループ用

    mov(r3,0)
    ldi(null,mask(A_ADDR),set_flags=True)
    ldi(r3,64*4,cond='zs')
    ldi(null,mask(Q_UNI_ITER),set_flags=True)
    mov(r3,-1,cond='zs')

    ldi(r1,16*4)

    L.STEP
    mov(uniforms_address,r2)
    nop()
    nop()
    ldi(ra0,64)
    
    L.UNI
    mov(broadcast,uniform)

    for i in range(7):
        mov(tmu0_s,r0)
        iadd(r0,r1,r0)
        nop(sig='load tmu0')
        fmul(ra1,r5,r4)
        nop()
        fadd(rb[i],ra1,rb[i])

    
    isub(ra0,ra0,1)
    jzc(L.UNI)
    nop()
    nop()
    nop()
    
    
    iadd(r2,r2,r3)
    jzc(L.STEP)
    nop()
    nop()
    nop()
    


    rotate(broadcast,r2,-Q_UNI_MOD)
    ldi(null,mask(Q_UNI_MOD),set_flags=True)
    mov(r2,0,cond='zs')
    isub(ra10,r5,0)
    jzc(L.UNI)
    mov(ra0,r5)
    ldi(null,mask(Q_UNI_ITER),set_flags=True)
    iadd(r2,r2,1,cond='zs')

    
    mutex_acquire() #排他制御 vpmは一つのため

    
    rotate(broadcast,r2,-THR_ID)
    isub(null,r5,1,set_flags=True)
    jzc(L.VPM)
    nop()
    nop()
    nop()

    rotate(broadcast,r2,-C_ADDR)
    imul24(r0,element_number,4)
    iadd(r0,r5,r0)
    for i in range(7):
        mov(tmu0_s,r0)
        iadd(r0,r1,r0)
        nop(sig='load tmu0')
        fadd(rb[i],rb[i],r4)

    L.VPM
    
    setup_dma_load_stride(16*4)
    setup_dma_load(mode='32bit horizontal', Y=0, nrows=7, mpitch=0)
    rotate(broadcast,r2,-OUT_ADDR)
    start_dma_load(r5)
    rotate(broadcast,r2,-VPM_FORM)
    wait_dma_load()

    setup_vpm_read(mode='32bit horizontal',Y=0,X=0,nrows=7) #loadしたDMAをvpmにread
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0) #書き込めるようにする

    
    for i in range(7):
        fmul(r0,vpm,r5)
        fadd(vpm,rb[i],r0)

    """
    rotate(broadcast,r2,-RELU_FLAG)
    isub(r3,r5,0)
    jzc(L.relu)
    nop()
    nop()
    nop()
    fmax(r0,r0,0.0)
    L.relu    
    mov(vpm,r0)
    """

    
    rotate(broadcast,r2,-OUT_ADDR)
    setup_dma_store(mode='32bit horizontal',nrows=7)
    start_dma_store(r5)
    wait_dma_store()

    mutex_release()
        


    
#====semafo=====    　すべてのスレッドが終わるまで待つ　詳細はquita見て
    sema_up(COMPLETED)
    rotate(broadcast,r2,-THR_ID)
    iadd(null,r5,-1,set_flags=True)
    jzc(L.skip_fin)
    nop()
    nop()
    nop()
    rotate(broadcast,r2,-THR)
    iadd(r1, r5, -1,set_flags=True)
    L.sem_down
    jzc(L.sem_down)
    sema_down(COMPLETED)    # Wait completion of all threads.
    nop()
    iadd(r1, r1, -1)
    
    interrupt()
    
    L.skip_fin
    exit(interrupt=False)
def GPU_dot(col,col_W,b,Relu_flag):
#def main():
    with Driver() as drv:
        SIMD=16
        UNIFORM=64
        n_threads=12
        p,q=col.shape
        r=col_W.shape[1]
        cal_q=q
        cal_r=r
        #rとqの調整
        rmod=r%SIMD
        if rmod!=0:
            cal_r+=SIMD-rmod
        qmod=q%n_threads
        if qmod!=0:
            cal_q+=n_threads-qmod

        q_th=int(cal_q/n_threads) #1thあたりのqの担当量    
        q_uni_iter=int(cal_q/n_threads/UNIFORM) #uniformの繰り返し回数
        q_uni_mod=int(cal_q/n_threads%UNIFORM) #uniformのあまり分
        r_simd_iter=int(cal_r/SIMD)
        print("q_th",q_th)
        print("q_uni_iter",q_uni_iter)
        print("uni_mod",q_uni_mod)
        print("r_simd_iter",r_simd_iter)


        
        vpm_form=1.0
        A=drv.alloc((p,cal_q),'float32')
        B=drv.alloc((cal_q,cal_r),'float32')
        C=drv.alloc((p,cal_r),'float32')
        out=drv.alloc((p,cal_r),'float32')
        out[:]=A[:]=B[:]=C[:]=0.0
        A[:]=col[:]
        B[:q,:r]=col_W[:]
        C[:,:r]=b[:]

        Relu_flag=1
        if(Relu_flag==0):
            CPUout=np.maximum(np.dot(A,B)+C,0.0)
        else:
            CPUout=np.dot(A,B)+C
        
        uniforms=drv.alloc((n_threads,16),'uint32')
        for th in range(n_threads):
            uniforms[th,0]=A.addresses()[0,int(th*q_th)]
            uniforms[th,1]=B.addresses()[int(th*q_th),0]
            uniforms[th,2]=out.addresses()[0,0]
            uniforms[th,3]=C.addresses()[0,0]
        uniforms[:,4]=q_uni_iter
        uniforms[:,5]=q_uni_mod
        uniforms[:,6]=r_simd_iter
        uniforms[:,7]=np.arange(1,(n_threads+1))
        uniforms[:,8]=n_threads
        uniforms[:,9]=B.strides[0]
        uniforms[:,10]=struct.unpack('L',struct.pack('f',vpm_form))[0]
        uniforms[:,11]=int(Relu_flag)
        code=drv.program(dot)
        elapsed_gpu=0
        start = time.time()
        
        drv.execute(
            n_threads=n_threads,
            program=code,
            uniforms=uniforms
        )
        
        elapsed_gpu += time.time() - start
        print ("elapsed_time:{:.4g}".format(elapsed_gpu*1000) + "[msec]")
        print('minimum absolute error: {:.4e}'.format(
            float(np.min(np.abs(CPUout[:,:r] - out[:,:r])))))
        print('maximum absolute error: {:.4e}'.format(
            float(np.max(np.abs(CPUout[:,:r] - out[:,:r])))))
        CPUout[:]=out[:]
        return CPUout