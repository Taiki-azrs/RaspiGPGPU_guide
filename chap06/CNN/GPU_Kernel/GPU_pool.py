# coding:utf-8
import numpy as np
import time
import math
import struct
from videocore.assembler import qpu
from videocore.driver import Driver
from tools.functions import *
from tools.util import im2col, col2im


def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values
def mask2(idx1,idx2):
    values = [1]*16
    values[idx1] = 0
    values[idx2] = 0
    return values
@qpu
def pool(asm,H,W,C,stride):
    X_ADDR=1
    OUT_ADDR=2
    THR_ID=3
    THR_NM=4
    COMPLETED = 0
    
    #set uniform to r2
    mov(r2,1)
    ldi(null,mask(X_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(OUT_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_ID),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')
    #set uniform end


    for i in range(32):
        mov(ra[i],0)

    REGNUM=int(12*(C/16))
    for k in range(12):
        ldi(r0,k*C*stride*4)
        imul24(r1,element_number,4)
        rotate(broadcast,r2,-X_ADDR)
        iadd(r1,r5,r1)
        iadd(r1,r1,r0)
        mov(r3,r1)
        for c in range(int(C/16)):
            ldi(r0,c*16*4)
            iadd(r1,r3,r0)
            mov(r3,r1)
            for i in range(2):
                for j in range(2):
                    mov(tmu0_s,r1)
                    ldi(r0,C*4)
                    iadd(r1,r1,r0)
                    nop(sig='load tmu0')
                    fmax(ra[k*int(C/16)+c],ra[k*int(C/16)+c],r4)
                ldi(r0,(i+1)*W*C*4)
                iadd(r1,r3,r0)

        
                
    mutex_acquire()
    #setup_dma_store_stride(16*4)
    rotate(broadcast,r2,-OUT_ADDR)
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0) #書き込めるようにする
    for i in range(REGNUM):
        mov(vpm,ra[i])
    setup_dma_store(mode='32bit horizontal',nrows=REGNUM)
    start_dma_store(r5)
    iadd(broadcast,r5,r1)
    wait_dma_store()
    mutex_release()

   
#====semaphore=====    
    sema_up(COMPLETED)
    rotate(broadcast,r2,-THR_ID)
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
def GPU_pool(x,stride,pad):
#def main():
    with Driver() as drv:
        SIMD=16
        UNIFORM=64
        n_threads=12
        #N=1;C=30;H=24;W=24
        N,C,H,W=x.shape
        cal_C=C
        Cmod=C%SIMD
        if Cmod!=0:
            cal_C+=SIMD-Cmod
        FH=2;FW=2
        oH=int(H/FH);oW=int(W/FW)
        th_oH=int(oH/n_threads)
        th_iter=int((th_oH*oW)/SIMD)
        X=drv.alloc((N,H,W,cal_C),'float32')
        out=drv.alloc((1,oH,oW,cal_C),'float32')
        X[:]=0
        X[:,:,:,:C]=x.transpose(0,2,3,1)[:]
        """
        x=np.random.randn(N,cal_C,H,W)
        x=np.arange(N*cal_C*H*W).reshape(N,cal_C,H,W)
        X[:]=x.transpose(0,2,3,1)

        """
        cetime=0
        start=time.time()
        out_h = int(1 + (H - FH) / stride)
        out_w = int(1 + (W - FW) / stride)
        col = im2col(x, FH, FW, stride, pad)
        col = col.reshape(-1, FH*FW)
        arg_max = np.argmax(col, axis=1)
        CPUout = np.max(col, axis=1)
        CPUout = CPUout.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        cetime=time.time()-start
        
        uniforms=drv.alloc((n_threads,16),'uint32')
        for th in range(n_threads):
            uniforms[th,0]=X.addresses()[0,th*th_oH*stride,0,0]
            uniforms[th,1]=out.addresses()[0,th*th_oH,0,0]
        uniforms[:,2]=np.arange(1,(n_threads+1))
        uniforms[:,3]=n_threads
        code=drv.program(pool,H,W,cal_C,stride)

        getime=0
        start=time.time()
        drv.execute(
            n_threads=n_threads,
            program=code,
            uniforms=uniforms
        )
        getime=time.time()-start
        print("===========Pooling=============")
        print("x size:{0},stride:{1},pad:{2}".format(x.shape,stride,pad))
        print("CPU time:{:.4f}[msec]".format(cetime*1000))
        print("GPU time:{:.4f}[msec]".format(getime*1000))
        """
        print("GPU time:{0}".format(etime*1000),"[msec]")
        print("CPU time:{0}".format(cpuetime*1000),"[msec]")
        """
        out_r=np.zeros((1,C,oH,oW))
        out_r[:]=out.transpose(0,3,1,2)[:,:C,:,:]

        print('minimum absolute error: {:.4e}'.format(
            float(np.min(np.abs(CPUout[:] - out_r[:])))))
        print('maximum absolute error: {:.4e}'.format(
            float(np.max(np.abs(CPUout[:] - out_r[:])))))
        """
        print('minimum relative error: {:.4e}'.format(
                float(np.min(np.abs((CPUout - out_r) / CPUout)))))
        print('maximum relative error: {:.4e}'.format(
                float(np.max(np.abs((CPUout - out_r) / CPUout)))))

        print("GPU{0}".format(out_r))
        print("CPU{0}".format(CPUout))
        """
        return out_r

