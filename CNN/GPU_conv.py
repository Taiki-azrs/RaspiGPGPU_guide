# coding:utf-8
#TODO バイアスの加算をuniformでチャレンジ
import numpy as np
import time
import math
import struct
from videocore.assembler import qpu
from videocore.driver import Driver
from tools.functions import *
from tools.util import im2col, col2im
from GPU_sgemm import sgemm


def mask(idx):
    values = [1]*16
    values[idx] = 0
    return values
@qpu
def conv(asm): #test
    CONVW_ADDR=0
    CONVX_ADDR=1
    CONVOUT_ADDR=2
    CB_ADDR=3
    SIMD_ITER=4
    TH_OH=5
    STR=6
    THR_ID=7
    THR_NM=8
    RELU=9
    W_BACKUP=10
    COMPLETED = 0
    
    #set uniform to r2
    mov(r2,1)
    ldi(null,mask(CONVW_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(CONVX_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(CONVOUT_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(CB_ADDR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(SIMD_ITER),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(TH_OH),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(STR),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_ID),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(RELU),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(W_BACKUP),set_flags=True)
    rotate(broadcast,r2,-CONVW_ADDR)
    mov(r2,r5,cond='zs')
    #set uniform end
    
    for i in range(32):
        mov(rb[i],0.0)
        mov(ra[i],0.0)
        
    renum=int(16/2)

    imul24(r0,element_number,4)
    rotate(broadcast,r2,-CONVX_ADDR)
    iadd(r0,r5,r0)
    L.loop
    ldi(r3,16*4)
    for n in range(int(64/16)):


        if(n*16==32):
            ldi(r1,4*4)
            iadd(r0,r0,r1)

        for i in range(5):
            for j in range(5):
                mov(uniforms_address,r2)
                mov(tmu0_s,r0)
                iadd(r0,r0,4)
                nop(sig='load tmu0')
                for k in range(renum*n,renum*(n+1)):
                    fmul(r1,r4,uniform)
                    fadd(ra[k],ra[k],r1)

                    fmul(r1,r4,uniform)
                    fadd(rb[k],rb[k],r1)

                ldi(null,mask(CONVW_ADDR),set_flags=True)
                iadd(r2,r2,r3,cond='zs')

            nop()
            rotate(broadcast,r2,-STR)
            iadd(r0,r0,r5)
            ldi(r1,5*4)
            isub(r0,r0,r1)

        ldi(r1,5)
        imul24(r1,r1,r5)
        isub(r0,r0,r1)
        iadd(r0,r0,r3)

        
        rotate(broadcast,r2,-W_BACKUP)
        ldi(null,mask(CONVW_ADDR),set_flags=True)
        mov(r2,r5,cond='zs')

    #set b
    rotate(uniforms_address,r2,-CB_ADDR)


    
    mutex_acquire()
    #setup_dma_store_stride(16*4)
    rotate(broadcast,r2,-CONVOUT_ADDR)
    setup_vpm_write(mode='32bit vertical',Y=0,X=0) #書き込めるようにする
    ldi(r1,16*16*4)
    for i in range(8):
        fadd(vpm,ra[i],uniform)
        mov(ra[i],0.0)
        fadd(vpm,rb[i],uniform)
        mov(rb[i],0.0)
    rotate(uniforms_address,r2,-CB_ADDR)
    setup_dma_store(mode='32bit horizontal',nrows=16)
    start_dma_store(r5)
    iadd(broadcast,r5,r1)
    for i in range(1,4):
        setup_vpm_write(mode='32bit vertical',Y=16*i,X=0)
        for j in range(8*i,8*(i+1)):
            fadd(vpm,ra[j],uniform)
            mov(ra[j],0.0)
            fadd(vpm,rb[j],uniform)
            mov(rb[j],0.0)
        rotate(uniforms_address,r2,-CB_ADDR)
        wait_dma_store()
        setup_dma_store(mode='32bit horizontal',Y=16*i,nrows=16)
        mov(vpm_st_addr,r5)
        iadd(broadcast,r5,r1)
    wait_dma_store()
    mutex_release()

    
    ldi(null,mask(SIMD_ITER),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.loop)
    ldi(r1,16*64*4)
    ldi(null,mask(CONVOUT_ADDR),set_flags=True)
    iadd(r2,r2,r1,cond='zs')
    
#====semafo=====    　すべてのスレッドが終わるまで待つ　詳細はquita見て
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
#def GPU_dot(col,col_W,b,Relu_flag=0):
def main():
    with Driver() as drv:
        SIMD=16
        UNIFORM=64
        n_threads=12
        N=1;C=1;H=28;W=36#28
        FN=16;FH=5;FW=5
        oH=H-int(FH/2)*2;oW=W-int(FW/2)*2
        th_oH=int(oH/n_threads)
        th_iter=int((th_oH*oW)/64)
        convX=drv.alloc((N,C,H,W),'float32')
        convW=drv.alloc((C,FH,FW,FN),'float32')
        convout=drv.alloc((C,oH,oW,FN),'float32')
        cb=drv.alloc(FN,'float32')
        convout[:]=0
        pad=0
        stride=1

        col=np.random.randn(N,C,H,W)
        col_W=np.random.randn(FN,C,FH,FW)
        b=np.random.randn(FN)
        #b[:]=1
        #b[16:]=0
        #col[:]=0
        #col_W[:]=0
        #col[:]=np.arange(N*C*H*W).reshape(N,C,H,W)
        #col_W[:]=np.arange(16*25).reshape(16,1,5,5)
        convX[:]=col[:]
        convW[:]=col_W.transpose(1,2,3,0)[:]
        out=np.zeros((768,16))
        cb[:]=b[:]
        
        #CPU time
        cpuetime=0
        start=time.time()
        out_h = 1 + int((H + 2*pad - FH) / stride)
        out_w = 1 + int((W + 2*pad - FW) / stride)
        col = im2col(col, FH, FW, stride, pad)
        col_W = col_W.reshape(FN, -1).T
        out = np.dot(col, col_W)+b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        cpuetime=time.time()-start
        
        CPU=out.transpose(0,2,3,1)
        
        #out = sgemm(col,col_W,b)
        
        uniforms=drv.alloc((n_threads,16),'uint32')
        uniforms[:,0]=convW.addresses()[0,0,0,0]
        for th in range(n_threads):
            uniforms[th,1]=convX.addresses()[0,0,th*th_oH,0]
            uniforms[th,2]=convout.addresses()[0,th*th_oH,0,0]
        uniforms[:,3]=cb.addresses()[0]
        uniforms[:,4]=th_iter
        uniforms[:,5]=th_oH
        uniforms[:,6]=int(W*4)
        uniforms[:,7]=np.arange(1,(n_threads+1))
        uniforms[:,8]=n_threads
        code=drv.program(conv)

        etime=0
        start=time.time()
        drv.execute(
            n_threads=n_threads,
            program=code,
            uniforms=uniforms
        )
        etime=time.time()-start
        
        print("GPU time:{0}".format(etime*1000),"[msec]")
        print("CPU time:{0}".format(cpuetime*1000),"[msec]")
        print('minimum absolute error: {:.4e}'.format(
            float(np.min(np.abs(CPU[:] - convout[:])))))
        print('maximum absolute error: {:.4e}'.format(
            float(np.max(np.abs(CPU[:] - convout[:])))))
        print('minimum relative error: {:.4e}'.format(
                float(np.min(np.abs((CPU - convout) / CPU)))))
        print('maximum relative error: {:.4e}'.format(
                float(np.max(np.abs((CPU - convout) / CPU)))))


        #print("GPU:{0}".format(convout[:]))
        #print("CPU:{0}".format(CPU))

main()
