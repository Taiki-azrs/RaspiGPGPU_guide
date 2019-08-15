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
def shockfilter(asm): #test
    A_ADDR=0
    B_ADDR=1
    C_ADDR=2
    R_DEV=3
    P=4
    Q=5
    R=6
    THR_NM=7
    THR=8
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
    ldi(null,mask(R_DEV),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(P),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(Q),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(R),set_flags=True)
    mov(r2,uniform,cond='zs')    
    ldi(null,mask(THR_NM),set_flags=True)
    mov(r2,uniform,cond='zs')
    ldi(null,mask(THR),set_flags=True)
    mov(r2,uniform,cond='zs')
    
    ldi(null,mask(UNI_ADDR),set_flags=True)
    mov(r2,r0,cond='zs')
    #set uniform end

    mov(uniforms_address,r2)
    ##################
    #ここまで
    

    ldi(ra21,tH)
    ldi(rb21,tH)
    rotate(broadcast,r2,-STR)
    mov(ra23,r5)
    rotate(broadcast,r2,-FILT)
    imul24(r0,element_number,4)
    
    iadd(r0,r5,r0)
    mov(tmu0_s,r0)
    ldi(ra22,640*4)
    nop(sig='load tmu0')
    mov(r0,r4) #filter係数
    ldi(ra24,16*4)
    
    L.wh
    #最初の画素計算
    isub(r1,rb21,ra21)
    imul24(r1,r1,ra23)
    imul24(r3,element_number,4)
    iadd(r1,r1,r3)
    
    rotate(broadcast,r2,-SRC)
    #上最初
    iadd(r1,r5,r1)
    mov(tmu0_s,r1)
    iadd(rb22,r1,ra24)
    nop(sig='load tmu0')
    mov(ra27,r4)

    #真ん中最初
    iadd(r1,r1,ra23)
    mov(tmu0_s,r1)
    iadd(rb23,r1,ra24)
    nop(sig='load tmu0')
    mov(ra28,r4)

    #下
    iadd(r1,r1,ra23)
    mov(tmu0_s,r1)
    iadd(rb24,r1,ra24)
    nop(sig='load tmu0')
    mov(ra29,r4)
    

    
    #レジスタ使用済み
    #ra23=STR  ra24=4*16
    #ra21=loop count rb21=base loop count
    #rb22=縦   ra22=アドレス加算用16*40
    #r1=対象画素  r0=filter
    #ra27=上最初 ra28=真ん中最初 ra29=した最初
    #rb22=上 rb23=真ん中 rb24=下:::アドレス
    for jj in range(3):
        mov(r1,ra27)
        for j in range(20): #1行分ra,rbに格納
            mov(tmu0_s,rb22)
            iadd(rb22,rb22,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左上
            mov(broadcast,r0)
            fmul(ra[j],r5,r1)

            #上
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-1)
            fmul(r3,r1,r5)
            fadd(ra[j],ra[j],r3)
            
            #右上
            rotate(r1,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-2)
            fmul(r3,r1,r5)
            fadd(ra[j],ra[j],r3)
            mov(r1,ra25)

            
            mov(tmu0_s,rb22)
            iadd(rb22,rb22,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左上
            mov(broadcast,r0)
            fmul(rb[j],r5,r1)

            #上
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-1)
            fmul(r3,r1,r5)
            fadd(rb[j],rb[j],r3)
            
            #右上
            rotate(r1,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-2)
            fmul(r3,r1,r5)
            fadd(rb[j],rb[j],r3)
            mov(r1,ra25)
            
        mov(ra27,r1) #次の上ループ用

        """
        mov(tmu0_s,ra26)
        mov(r1,ra26)
        iadd(rb23,r1,ra24)
        iadd(r1,r1,ra23)
        nop(sig='load tmu0')
        mov(ra26,r1)
        mov(r1,r4)
        """
        mov(r1,ra28)
        for j in range(20):
            mov(tmu0_s,rb23)
            iadd(rb23,rb23,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左
            rotate(broadcast,r0,-3)
            #fmul(ra[j],r5,r1)
            fmul(r3,r5,r1)
            fadd(ra[j],ra[j],r3)
            
            #真ん中
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-4)
            fmul(r3,r1,r5)
            fadd(ra[j],ra[j],r3)
            
            
            #右
            rotate(r1,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-5)
            fmul(r3,r1,r5)
            fadd(ra[j],ra[j],r3)
            mov(r1,ra25)

            mov(tmu0_s,rb23)
            iadd(rb23,rb23,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左
            rotate(broadcast,r0,-3)
            #fmul(rb[j],r5,r1)
            fmul(r3,r5,r1)
            fadd(rb[j],rb[j],r3)
            
            #真ん中
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-4)
            fmul(r3,r1,r5)
            fadd(rb[j],rb[j],r3)
            
            #右
            rotate(r1,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-5)
            fmul(r3,r1,r5)
            fadd(rb[j],rb[j],r3)
            mov(r1,ra25)

        mov(ra28,r1)
        """"
        mov(tmu0_s,ra26)
        mov(r1,ra26)
        iadd(rb24,r1,ra24)
        nop(sig='load tmu0')
        mov(r1,r4)
        """
        mov(r1,ra29)
        for j in range(20):
            mov(tmu0_s,rb24)
            iadd(rb24,rb24,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左下
            rotate(broadcast,r0,-6)
            #fmul(ra[j],r5,r1)
            fmul(r3,r5,r1)
            fadd(ra[j],ra[j],r3)
            
            #下
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-7)
            fmul(r3,r1,r5)
            fadd(ra[j],ra[j],r3)
            
            #右下
            rotate(r1,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-8)
            fmul(r3,r1,r5)
            fadd(ra[j],ra[j],r3)
            mov(r1,ra25)

            mov(tmu0_s,rb24)
            iadd(rb24,rb24,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左下
            rotate(broadcast,r0,-6)
            #fmul(rb[j],r5,r1)
            fmul(r3,r5,r1)
            fadd(rb[j],rb[j],r3)

            #下
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-7)
            fmul(r3,r1,r5)
            fadd(rb[j],rb[j],r3)
            
            #右下
            rotate(r1,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            rotate(broadcast,r0,-8)
            fmul(r3,r1,r5)
            fadd(rb[j],rb[j],r3)
            mov(r1,ra25)
        mov(ra29,r1)

        mutex_acquire() #排他制御 vpmは一つのため

        setup_vpm_read(mode='32bit horizontal',Y=0,X=0,nrows=40) #loadしたDMAをvpmにread
        setup_vpm_write(mode='32bit horizontal',Y=0,X=0) #書き込めるようにする
        for kj in range(20):  #ra,rbをvpmにmov
            mov(vpm,ra[kj])
            mov(vpm,rb[kj])
        
        rotate(broadcast,r2,-GAUS)
        setup_dma_store(mode='32bit horizontal',nrows=40)
        start_dma_store(r5)
        ldi(null,mask(GAUS),set_flags=True)
        iadd(r2,r5,ra22,cond='zs')
        wait_dma_store()
        
        mutex_release()
        
    isub(ra21,ra21,1)
    jzc(L.wh)
    nop()
    nop()
    nop()
    nop()

    
    ####次
    ldi(rb30,0.25)#dt
    mov(ra21,rb21)
    nop()
    L.wh2
    isub(r1,rb21,ra21)
    imul24(r1,r1,ra23)
    imul24(r3,element_number,4)
    iadd(r1,r1,r3)
    rotate(broadcast,r2,-GAUS_BASE)
    #上最初
    iadd(r1,r5,r1)
    mov(tmu0_s,r1)
    iadd(rb22,r1,ra24)
    nop(sig='load tmu0')
    mov(ra27,r4)

    #真ん中
    iadd(r1,r1,ra23)
    mov(tmu0_s,r1)
    iadd(rb23,r1,ra24)
    nop(sig='load tmu0')
    mov(ra28,r4)

    #下
    iadd(r1,r1,ra23)
    mov(tmu0_s,r1)
    iadd(rb24,r1,ra24)
    nop(sig='load tmu0')
    mov(ra29,r4)

    for jj in range(3):
        for j in range(20):
            mov(r1,ra27)
            mov(tmu0_s,rb22)
            iadd(rb22,rb22,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #上
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            mov(ra[j],r1)
            mov(ra27,ra25)

            #真ん中
            mov(r1,ra28)
            mov(tmu0_s,rb23)
            iadd(rb23,rb23,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左
            fadd(ra[j],ra[j],r1)
            #真ん中
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            mov(r0,r1)
            ldi(r3,-4.0)
            fmul(r3,r1,r3)
            fadd(ra[j],ra[j],r3)
            
            #右
            rotate(r3,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r3,r5,cond='zs')
            fadd(ra[j],ra[j],r3)
            mov(ra28,ra25)

            #norm
            fsub(ra30,r3,r0)
            mov(r3,r0)
            #fsub(r3,ra29,r1)
            fmul(ra30,ra30,ra30)
            
            #下
            mov(r1,ra29)
            mov(tmu0_s,rb24)
            iadd(rb24,rb24,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            fadd(ra[j],ra[j],r1)

            #norm i+1-i
            fsub(r3,r1,r0)
            fmul(r3,r3,r3)
            fadd(ra30,ra30,r3)
            nop()
            mov(sfu_recipsqrt,ra30)
            nop()
            nop()
            fmul(ra30,r4,ra30)

            mov(ra29,ra25)

            ldi(r3,0.25)
            fmul(ra[j],ra[j],r3)
            
            ldi(r3,-1.0)
            fmax(ra[j],ra[j],r3)
            ldi(r3,1.0)
            fmin(ra[j],ra[j],r3)
            fmul(r3,ra30,rb30)
            fmul(r3,r3,ra[j])
            fsub(ra[j],r0,r3)

            
            mov(r1,ra27)
            mov(tmu0_s,rb22)
            iadd(rb22,rb22,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #上
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            mov(rb[j],r1)
            mov(ra27,ra25)

            #真ん中
            mov(r1,ra28)
            mov(tmu0_s,rb23)
            iadd(rb23,rb23,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            #左
            fadd(rb[j],rb[j],r1)
            #真ん中
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            mov(r0,r1)
            ldi(r3,-4.0)
            fmul(r3,r1,r3)
            fadd(rb[j],rb[j],r3)            
            #右
            rotate(r3,r1,-1)
            rotate(broadcast,ra25,-1)
            ldi(null,mask(15),set_flags=True)
            mov(r3,r5,cond='zs')
            fadd(rb[j],rb[j],r3)
            mov(ra28,ra25)

            #norm
            fsub(ra30,r3,r1)
            mov(r3,r1)
            #fsub(r3,ra29,r1)
            fmul(ra30,ra30,ra30)
        
            #下
            mov(r1,ra29)
            mov(tmu0_s,rb24)
            iadd(rb24,rb24,ra24)
            nop(sig='load tmu0')
            mov(ra25,r4) #次の画素
            rotate(r1,r1,-1)
            mov(broadcast,ra25)
            ldi(null,mask(15),set_flags=True)
            mov(r1,r5,cond='zs')
            fadd(rb[j],rb[j],r1)
            
            fsub(r3,r1,r3)
            fmul(r3,r3,r3)
            fadd(ra30,ra30,r3)
            nop()
            mov(sfu_recipsqrt,ra30)
            nop()
            nop()
            fmul(ra30,r4,ra30)

            ldi(r3,0.25)
            fmul(rb[j],rb[j],r3)
            
            mov(ra29,ra25)
            ldi(r3,-1.0)
            fmax(rb[j],rb[j],r3)
            ldi(r3,1.0)
            fmin(rb[j],rb[j],r3)
            fmul(r3,ra30,rb30)
            fmul(r3,r3,rb[j])
            fsub(rb[j],r0,r3)
            
        mutex_acquire() #排他制御 vpmは一つのため

        setup_vpm_read(mode='32bit horizontal',Y=0,X=0,nrows=40) #loadしたDMAをvpmにread
        setup_vpm_write(mode='32bit horizontal',Y=0,X=0) #書き込めるようにする
        for kj in range(20):  #ra,rbをvpmにmov
            mov(vpm,ra[kj])
            mov(vpm,rb[kj])
        
        rotate(broadcast,r2,-SHK)
        setup_dma_store(mode='32bit horizontal',nrows=40)
        start_dma_store(r5)
        ldi(null,mask(SHK),set_flags=True)
        iadd(r2,r5,ra22,cond='zs')
        wait_dma_store()
        
        mutex_release()
        
    isub(ra21,ra21,1)
    jzc(L.wh2)
    nop()
    nop()
    nop()

            
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
    
with Driver() as drv:
    p=1
    q=4320
    r=100
    A=drv.alloc((p,q),'float32')
    B=drv.alloc((q,r),'float32')
    C=drv.alloc((p,r),'float32')
    n_threads=10
    r_dev=r/n_threads
    
    global Iter_W
    Iter_W=math.ceil(r_dev/16)
    
    uniforms=drv.alloc((n_threads,10),'uint32')
    uniforms[:,0]=uniforms.addresses()[:,0]
    for th in range(n_threads):
        uniforms[th,1]=A.addresses()[0,0]
        uniforms[th,2]=B.addresses()[0,int(th*r_dev)]
        uniforms[th,3]=C.addresses()[0,int(th*r_dev)]
        
    uniforms[:,4]=r_dev
    uniforms[:,5]=p
    uniforms[:,6]=q
    uniforms[:,7]=r
    #uniforms[:,5]=A.strides[0] #WIDTH  use address 1=8bit 32bit=4 4*WIDTH
    uniforms[:,8]=np.arange(1,(n_threads+1))
    uniforms[:,9]=n_threads
    code=drv.program(shockfilter)
    elapsed_gpu=0
    for i in range(100):
        start = time.time()
        drv.execute(
            n_threads=n_threads,
            program=code,
            uniforms=uniforms
        )
        elapsed_gpu += time.time() - start
    print ("elapsed_time:{0}".format(elapsed_gpu/100) + "[sec]")
    
