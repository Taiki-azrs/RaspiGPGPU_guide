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
def conv(asm,H,W,FH,FW,FN,oH,oW): 
    CONVW_ADDR=0
    CONVX_ADDR=1
    CONVOUT_ADDR=2
    CB_ADDR=3
    SIMD_ITER=4
    TH_OH=5
    STR=6
    C_ITER=7
    THR_ID=8
    THR_NM=9
    RELU=10
    W_BACKUP=11
    X_BACKUP=12
    OUT_BACKUP=13
    SITER_BACKUP=14
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
    ldi(null,mask(C_ITER),set_flags=True)
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
    
    ldi(null,mask(X_BACKUP),set_flags=True) #Backup
    rotate(broadcast,r2,-CONVX_ADDR)
    mov(r2,r5,cond='zs')
    
    ldi(null,mask(OUT_BACKUP),set_flags=True)
    rotate(broadcast,r2,-CONVOUT_ADDR)
    mov(r2,r5,cond='zs')
    
    ldi(null,mask(SITER_BACKUP),set_flags=True)
    rotate(broadcast,r2,-SIMD_ITER)
    mov(r2,r5,cond='zs')
    
    #set uniform end

    for i in range(32): #初期化
        mov(rb[i],0.0)
        mov(ra[i],0.0)
    L.cloop
    renum=int(FN/2)
    imul24(r0,element_number,4)
    rotate(broadcast,r2,-CONVX_ADDR)
    iadd(r0,r5,r0)
    L.simdloop


    for n in range(int(64/FN)):

        for i in range(FH):
            for j in range(FW):
                mov(uniforms_address,r2)
                mov(tmu0_s,r0)
                iadd(r0,r0,4)
                nop(sig='load tmu0')
                for k in range(renum*n,renum*(n+1)):
                    fmul(r1,r4,uniform)
                    fadd(ra[k],ra[k],r1)

                    fmul(r1,r4,uniform)
                    fadd(rb[k],rb[k],r1)

                ldi(r3,FN*4)
                ldi(null,mask(CONVW_ADDR),set_flags=True)
                iadd(r2,r2,r3,cond='zs')

            rotate(broadcast,r2,-STR)
            iadd(r0,r0,r5)
            ldi(r1,FW*4)
            isub(r0,r0,r1)

        ldi(r1,FW)
        imul24(r1,r1,r5)
        isub(r0,r0,r1)
        ldi(r3,16*4)
        iadd(r0,r0,r3)

        
        rotate(broadcast,r2,-W_BACKUP)
        ldi(null,mask(CONVW_ADDR),set_flags=True)
        mov(r2,r5,cond='zs')
        
        if((n+1)*16==int(oW)): #端処理
            ldi(r1,(int(FW/2))*2*4)
            iadd(r0,r0,r1)

    #set b
    rotate(broadcast,r2,-C_ITER)
    isub(null,r5,1)
    jzc(L.add_b)
    nop()
    nop()
    nop()
    for i in range(int(64/FN)):
        rotate(uniforms_address,r2,-CB_ADDR)
        nop();nop()
        for j in range(int(FN/2)):
            idx=int(j+(i*FN/2))
            fadd(ra[idx],ra[idx],uniform)
            fadd(rb[idx],rb[idx],uniform)
    L.add_b

    mutex_acquire()
    setup_dma_load_stride(FN*4)
    setup_dma_store_stride((FN-16)*4)
    rotate(broadcast,r2,-CONVOUT_ADDR)
    rfn=int(64/FN)
    simfn=int(FN/16)
    for m in range(int(rfn)):
        for n in range(int(simfn)):
            setup_dma_load(mode='32bit horizontal', Y=16*(m*simfn+n), nrows=16, mpitch=0)
            start_dma_load(r5)
            wait_dma_load()
            setup_vpm_read(mode='32bit vertical', Y=16*(m*simfn+n), X=0, nrows=16)
            setup_vpm_write(mode='32bit vertical',Y=16*(m*simfn+n),X=0) 
            for i in range(8*(m*simfn+n),8*((m*simfn+n)+1)):
                fadd(vpm,vpm,ra[i])
                fadd(vpm,vpm,rb[i])
                mov(ra[i],0.0)
                mov(rb[i],0.0)
            setup_dma_store(mode='32bit horizontal',Y=16*(simfn*m+n),nrows=16)
            start_dma_store(r5)
            ldi(r1,16*4)
            iadd(broadcast,r5,r1)
            wait_dma_store()
        ldi(r1,16*FN*4-(FN*4))
        iadd(broadcast,r5,r1)
    mutex_release()

    ldi(null,mask(SIMD_ITER),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.simdloop)
    ldi(r1,16*64*4)
    ldi(null,mask(CONVOUT_ADDR),set_flags=True)
    iadd(r2,r2,r1,cond='zs')

    
    rotate(broadcast,r2,-SITER_BACKUP)
    ldi(null,mask(SIMD_ITER),set_flags=True)
    mov(r2,r5,cond='zs')
    
    
    rotate(broadcast,r2,-OUT_BACKUP)
    ldi(null,mask(CONVOUT_ADDR),set_flags=True)
    mov(r2,r5,cond='zs')

    
    ldi(r1,H*W*4)
    rotate(broadcast,r2,-X_BACKUP)
    ldi(null,mask2(CONVX_ADDR,X_BACKUP),set_flags=True)
    iadd(r2,r5,r1,cond='zs')
    
    
    ldi(r1,FH*FW*FN*4)
    rotate(broadcast,r2,-W_BACKUP)
    ldi(null,mask2(CONVW_ADDR,W_BACKUP),set_flags=True)
    iadd(r2,r5,r1,cond='zs')
    
    ldi(null,mask(C_ITER),set_flags=True) #Cloop
    isub(r2,r2,1,cond='zs')
    jzc(L.cloop)
    nop()
    nop()
    nop()
    nop()
    
    rotate(broadcast,r2,-RELU) #Relu
    isub(null,r5,1)
    jzc(L.relu_end)
    nop()
    nop()
    nop()
    
    imul24(r0,element_number,4)
    rotate(broadcast,r2,-CONVOUT_ADDR)
    iadd(r0,r5,r0)


    L.reluloop
    ldi(r3,16*4)
    for i in range(32):
        mov(tmu0_s,r0)
        iadd(r0,r0,r3)
        nop(sig='load tmu0')
        mov(r1,0.0)
        fmax(ra[i],r4,r1)

        
        mov(tmu0_s,r0)
        iadd(r0,r0,r3)
        nop(sig='load tmu0')
        mov(r1,0.0)
        fmax(rb[i],r4,r1)
    setup_dma_store_stride(0)
    mutex_acquire()
    setup_vpm_write(mode='32bit horizontal',Y=0,X=0)
    for i in range(32):
        mov(vpm,ra[i])
        mov(vpm,rb[i])
    setup_dma_store(mode='32bit horizontal',Y=0,nrows=64)
    start_dma_store(r5)
    wait_dma_store()
    ldi(r1,16*64*4)
    iadd(broadcast,r5,r1)
    mutex_release()
    ldi(null,mask(SIMD_ITER),set_flags=True)
    isub(r2,r2,1,cond='zs')
    jzc(L.reluloop)
    nop()
    nop()
    nop()
    
    L.relu_end
    
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
    sema_down(COMPLETED)    
    nop()
    iadd(r1, r1, -1)
    interrupt()
    L.skip_fin
    exit(interrupt=False)
def main():
    with Driver() as drv:
        class Color:
            BLACK     = '\033[30m'
            RED       = '\033[31m'
            GREEN     = '\033[32m'
            YELLOW    = '\033[33m'
            BLUE      = '\033[34m'
            PURPLE    = '\033[35m'
            CYAN      = '\033[36m'
            WHITE     = '\033[37m'
            END       = '\033[0m'
            BOLD      = '\038[1m'
            UNDERLINE = '\033[4m'
            INVISIBLE = '\033[08m'
            REVERCE   = '\033[07m'
        SIMD=16;UNIFORM=64;n_threads=12

        N=1;C=3;H=64;W=64
        FN=16;FH=5;FW=5
        Relu_flag=1
        x=np.random.randn(N,C,H,W)
        w=np.random.randn(FN,C,FH,FW)
        b=np.random.randn(FN)
        N,C,H,W=x.shape
        FN,C,FH,FW=w.shape
        calc_H=H;calc_W=W;calc_FN=FN
        eH=int(FH/2)*2;eW=int(FW/2)*2
        oH=H-eH;oW=W-eW
        modH=oH%n_threads;modW=oW%SIMD;modFN=FN%SIMD
        if(modH!=0):
            calc_H+=n_threads-modH
        if(modW!=0):
            calc_W+=SIMD-modW
        if(modFN!=0):
            calc_FN+=SIMD-modFN
        calc_oH=calc_H-eH
        calc_oW=calc_W-eW

        th_oH=int(calc_oH/n_threads)
        th_iter=int((th_oH*calc_oW)/(64/calc_FN*16))
        convX=drv.alloc((N,C,calc_H,calc_W),'float32')
        convW=drv.alloc((C,FH,FW,calc_FN),'float32')
        convout=drv.alloc((1,calc_oH,calc_oW,calc_FN),'float32')
        cb=drv.alloc(calc_FN,'float32')
        convout[:]=0;convX[:]=0;convW[:]=0;cb[:]=0
        pad=0
        stride=1
        
        convX[:,:,:H,:W]=x[:]
        convW[:,:,:,:FN]=w.transpose(1,2,3,0)[:]#転置してcopy
        cb[:FN]=b[:]
        

        
        uniforms=drv.alloc((n_threads,16),'uint32') 
        uniforms[:,0]=convW.addresses()[0,0,0,0]
        for th in range(n_threads):
            uniforms[th,1]=convX.addresses()[0,0,th*th_oH,0]
            uniforms[th,2]=convout.addresses()[0,th*th_oH,0,0]
        uniforms[:,3]=cb.addresses()[0]
        uniforms[:,4]=th_iter
        uniforms[:,5]=th_oH
        uniforms[:,6]=int(calc_W*4)
        uniforms[:,7]=C
        uniforms[:,8]=np.arange(1,(n_threads+1))
        uniforms[:,9]=n_threads
        uniforms[:,10]=Relu_flag+1
        code=drv.program(conv,calc_H,calc_W,FH,FW,calc_FN,calc_oH,calc_oW)#引数渡し

        while(1):
            #CPU Calculation
            #im2col->dot
            cpuetime=0
            start=time.time()
            out_h = 1 + int((H + 2*pad - FH) / stride)
            out_w = 1 + int((W + 2*pad - FW) / stride)
            col = im2col(x, FH, FW, stride, pad)
            col_W = w.reshape(FN, -1).T
            out = np.dot(col, col_W)+b
            #out = np.maximum(out,0.0)
            CPU = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
            cetime=time.time()-start
        
    
            start=time.time()
            drv.execute(
                n_threads=n_threads,
                program=code,
                uniforms=uniforms
            )
            getime=time.time()-start
            
            GPU=np.zeros((C,FN,oH,oW))
            tranout=convout.transpose(0,3,1,2)

            GPU[:]=tranout[:,:FN,:oH,:oW]
            print("===========畳み込み層=============")
            print("x size:{0},w size:{1}".format(x.shape,w.shape))
            print("CPU time:{:.4f}".format(cetime*1000),"[msec]")
            print("GPU time:{:.4f}".format(getime*1000),"[msec]")
            print('minimum absolute error: {:.4e}'.format(
                float(np.min(np.abs(CPU[:] - GPU[:])))))
            print('maximum absolute error: {:.4e}'.format(
                float(np.max(np.abs(CPU[:] - GPU[:])))))
            print(Color.GREEN+"{:.2f}倍高速化!!!".format(cetime/getime)+Color.END)
            convout[:]=0
            time.sleep(3)
main()
