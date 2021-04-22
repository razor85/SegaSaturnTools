; Copyright 2020-2021 Romulo Fernandes Machado Leitao <romulo@castorgroup.net>
; 
; Permission is hereby granted, free of charge, to any person obtaining a
; copy of this software and associated documentation files (the "Software"),
; to deal in the Software without restriction, including without limitation
; the rights to use, copy, modify, merge, publish, distribute, sublicense,
; and/or sell copies of the Software, and to permit persons to whom the
; Software is furnished to do so, subject to the following conditions:
; 
; The above copyright notice and this permission notice shall be included in
; all copies or substantial portions of the Software.
; 
; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
; IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
; FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
; AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
; LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
; FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
; DEALINGS IN THE SOFTWARE.

ORG 0
Start

  ; CT3:
  ;  0: Model data ptr 
  ;  1: input vertices ptr
  ;  2: output vertices ptr
  ;  3: camera view matrix ptr

  ; Negative camera position
  mov 27,CT0
  mov 4,CT3

  ; Copy 9 words
  ; Camera forward vector in model space
  ; Faces normals ptr, dot ptr, numfaces
  mov 8,LOP
  mov COPYINPUTS,TOP

COPYINPUTS: mov MC3,MC0
  btm
  nop

  ;
  ; Model Data:
  ;   matrix(9) + position(3) + numVertices(1) = 13
  mov 0,CT0
  mov 0,CT3
  mov MC3,A
  mov ALL,RA0
  dma D0,M0,13

  ; Wait for transfer
DMAMODEL: jmp T0,DMAMODEL
  nop

  mov 3,CT3
  mov MC3,A
  mov ALL,RA0
  mov 18,CT0
  dma D0,M0,9

  ; Wait for transfer
DMAMODEL2: jmp T0,DMAMODEL2
  nop

  ; CT0:
  ;  0: matrix(9)
  ;  9: position(3)
  ; 12: numVertices
  ; 13: inputOffset
  ; 14: outputOffset
  ; 15: pendingTransfer
  ; 16: pendingTransfer * 3
  ; 17: current processing vertex index at MC1
  ; 18: view matrix(9)
  ; 27: negative camera position(3)
  ; 30: camera forward modelspace(3)
  ; 33: faces normals ptr
  ; 34: faces dot ptr
  ; 35: numfaces
  ; 36: pending faces
  ; 37: pending faces * 3
 
  mov 13,CT0
  mov 1,CT3
  
  ; CT0[13] = CT3[1]; CT0 = 14;
  mov MC3,MC0

  ; CT0[14] = CT3[2]; CT0 = 15;
  mov MC3,MC0

  ; CT0[15] = 0; CT0[16] = 0;
  mvi 0,MC0
  mvi 0,MC0

  ; Copy up to 21 vertices to process
COPYVERTICES: mov 20,LOP
  mov COPYVERTICES_LOOP,TOP

  ; CT0[15] += 1
COPYVERTICES_LOOP: mov 15,CT0 clr A
  mov M0,A
  mvi 1,PL
  add mov ALL,MC0

  ; CT0[16] += 3
  mov 16,CT0 clr A
  mov M0,A
  mvi 3,PL
  add mov ALL,MC0

  ; CT0[12] -= 1
  mov 12,CT0 clr A
  mov M0,A
  mvi 1,PL
  sub mov ALL,MC0
  
  ; Do we have anything else to process?
  ; if (!CT0[12]) goto TRANSFERVERTICES;
  jmp Z,TRANSFERVERTICES
  nop

  btm
  nop
  
  ; copy up to CT0[16] vertices from CT0[13] 
  ; RA0 = CT0[13]; CT1 = 0; CT0 = 16;
TRANSFERVERTICES: mov 13,CT0 clr A
  mov M0,A
  mov ALL,RA0
  mov 16,CT0

  ; CT0[13] += CT0[16]
  mov MC0,PL
  mov 13,CT0
  add mov ALL,MC0

  mov 16,CT0
  mov 0,CT1
  dma D0,M1,M0

WAIT_ITRANSFER: jmp T0,WAIT_ITRANSFER
  nop

  ; Now we have up to CT0[15] vertices to process
  ; CT1 = 0; CT3 = 0
  mov 0,CT1
  mov PROCVERTEX,TOP

  ; LOP = CT0[15] - 1
  mov 15,CT0 clr A
  mov M0,A
  mvi 1,PL
  sub mov ALL,LOP
  mov 0,CT2
  mov 0,CT3
  
  ; Store current processed vertex at CT0[17]
  mov 17,CT0
  mvi 0,MC0

  ; 3x3 Matrix at CT0[0]
  ; 3x1 Vector at CT1[9]
PROCVERTEX: mov 0,CT0
  ; output.x =================================================================
  ; X = a[0]; Y = b[0]; A = 0;
  mov MC0,X mov MC1,Y clr A

  ; X = a[1]; Y = b[1]; P = a[0] * b[0];
  mov MC0,X mov MUL,P mov MC1,Y 

  ; A += P; X = a[2]; Y = b[2]; P = a[1] * b[1];
  ad2 mov MC0,X mov MUL,P mov MC1,Y mov ALU,A

  ; A += P; P = a[2] * b[2];
  ad2 mov MUL,P mov ALU,A
  
  ; A += P; PL = ALH
  ad2 mov ALU,A mov ALH,PL

  ; Sum position
  mov 9,CT0 clr A
  mov MC0,A
  add MOV ALU,A mov ALL,MC2

  ; output.y =================================================================
  mov 17,CT0
  mov M0,CT1
  mov 3,CT0

  ; X = a[0]; Y = b[0]; A = 0;
  mov MC0,X mov MC1,Y clr A

  ; X = a[1]; Y = b[1]; P = a[0] * b[0];
  mov MC0,X mov MUL,P mov MC1,Y 

  ; A += P; X = a[2]; Y = b[2]; P = a[1] * b[1];
  ad2 mov MC0,X mov MUL,P mov MC1,Y mov ALU,A

  ; A += P; P = a[2] * b[2];
  ad2 mov MUL,P mov ALU,A
  ad2 mov ALU,A mov ALH,PL

  ; Sum position
  mov 10,CT0 clr A
  mov MC0,A
  add MOV ALU,A mov ALL,MC2

  ; output.z =================================================================
  mov 17,CT0
  mov M0,CT1
  mov 6,CT0

  ; X = a[0]; Y = b[0]; A = 0;
  mov MC0,X mov MC1,Y clr A

  ; X = a[1]; Y = b[1]; P = a[0] * b[0];
  mov MC0,X mov MUL,P mov MC1,Y 

  ; A += P; X = a[2]; Y = b[2]; P = a[1] * b[1];
  ad2 mov MC0,X mov MUL,P mov MC1,Y mov ALU,A

  ; A += P; P = a[2] * b[2];
  ad2 mov MUL,P mov ALU,A
  ad2 mov ALU,A mov ALH,PL

  ; Sum position
  mov 11,CT0 clr A
  mov MC0,A
  add MOV ALU,A mov ALL,MC2

  ; ==========================================================================
  ; Start with camera matrix. We don't do camera position because divu is
  ; going to stall, so we do it in the SH2 instead.

  ; output.x =================================================================
  mov 17,CT0
  mov MC0,CT2

  ; X = a[0]; Y = b[0]; A = 0;
  mov MC0,X mov MC2,Y clr A

  ; X = a[1]; Y = b[1]; P = a[0] * b[0];
  mov MC0,X mov MUL,P mov MC2,Y 

  ; A += P; X = a[2]; Y = b[2]; P = a[1] * b[1];
  ad2 mov MC0,X mov MUL,P mov MC2,Y mov ALU,A

  ; A += P; P = a[2] * b[2];
  ad2 mov MUL,P mov ALU,A
  ad2 mov ALU,A mov ALH,MC3

  ; output.y =================================================================
  mov 17,CT0
  mov M0,CT2
  mov 21,CT0

  ; X = a[0]; Y = b[0]; A = 0;
  mov MC0,X mov MC2,Y clr A

  ; X = a[1]; Y = b[1]; P = a[0] * b[0];
  mov MC0,X mov MUL,P mov MC2,Y 

  ; A += P; X = a[2]; Y = b[2]; P = a[1] * b[1];
  ad2 mov MC0,X mov MUL,P mov MC2,Y mov ALU,A

  ; A += P; P = a[2] * b[2];
  ad2 mov MUL,P mov ALU,A
  ad2 mov ALU,A mov ALH,MC3

  ; output.z =================================================================
  mov 17,CT0
  mov M0,CT2
  mov 24,CT0

  ; X = a[0]; Y = b[0]; A = 0;
  mov MC0,X mov MC2,Y clr A

  ; X = a[1]; Y = b[1]; P = a[0] * b[0];
  mov MC0,X mov MUL,P mov MC2,Y 

  ; A += P; X = a[2]; Y = b[2]; P = a[1] * b[1];
  ad2 mov MC0,X mov MUL,P mov MC2,Y mov ALU,A

  ; A += P; P = a[2] * b[2];
  ad2 mov MUL,P mov ALU,A
  ad2 mov ALU,A mov ALH,MC3

  ; done vertex transformation ===============================================

  ; CT0[17] += 3
  mov 17,CT0 clr A
  mov M0,A
  mvi 3,PL
  add mov ALL,MC0

  ; Back to the loop
  btm
  nop

  ; Transfer up to CT0[16] vertices to CT0[14]
  mov 14,CT0 clr A
  mov M0,A
  mov ALL,WA0

  mov 16,CT0
  mov M0,P

  ; CT[14] += CT[16]
  mov 14,CT0
  add mov ALL,MC0

  ; Transfer out and reset CT0[15]
  mov 16,CT0
  mov 0,CT3
  dma M3,D0,M0

  ; CT0[15] = 0; CT0[16] = 0;
  mov 15,CT0
  mvi 0,MC0
  mvi 0,MC0

WAIT_OTRANSFER: jmp T0,WAIT_OTRANSFER
  nop

  ; if (CT0[12]) goto COPYVERTICES
  mov 12,CT0 clr A
  mov M0,A
  mvi 0,PL
  add
  
  jmp NZ,COPYVERTICES
  nop

  ; Compute faces dot product against camera forward =========================
  mov 36,CT0
  mvi 0,MC0
  mvi 0,MC0

  ; Copy up to 21 normals to process
COPYNORMALS: mov 20,LOP
  mov COPYNORMALS_LOOP,TOP

  ; CT0[36] += 1
  ; CT0[37] += 3
COPYNORMALS_LOOP: mov 36,CT0 clr A
  mov M0,A
  mvi 1,PL
  add mov ALL,MC0

  ; CT0[37] += 3
  mov 37,CT0 clr A
  mov M0,A
  mvi 3,PL
  add mov ALL,MC0

  ; CT0[35] -= 1
  mov 35,CT0 clr A
  mov M0,A
  mvi 1,PL
  sub mov ALL,MC0
  
  ; Do we have anything else to process?
  ; if (!CT0[35]) goto TRANSFERNORMALS;
  jmp Z,TRANSFERNORMALS
  nop

  btm
  nop

  ; copy up to CT0[36] vertices(CT0[37] words) from CT0[33] 
  ; RA0 = CT0[33]; CT1 = 0; CT0 = 37;
TRANSFERNORMALS: mov 33,CT0 clr A
  mov M0,A
  mov ALL,RA0
  mov 37,CT0

  ; CT0[33] += CT0[37]
  mov MC0,PL
  mov 33,CT0
  add mov ALL,MC0

  mov 37,CT0
  mov 0,CT1
  dma D0,M1,M0

WAIT_NITRANSFER: jmp T0,WAIT_NITRANSFER
  nop

  ;=========================================================================
  ; Now we have up to CT0[36] normals in M1 to process
  mov 0,CT1
  mov 0,CT2
  mov PROCNORMAL,TOP

  ; LOP = CT0[36] - 1
  mov 36,CT0 clr A
  mov M0,A
  mvi 1,PL
  sub mov ALL,LOP
  
PROCNORMAL: mov 30,CT0
  ; X = a[0]; Y = b[0]; A = 0;
  mov MC0,X mov MC1,Y clr A

  ; X = a[1]; Y = b[1]; P = a[0] * b[0];
  mov MC0,X mov MUL,P mov MC1,Y 

  ; A += P; X = a[2]; Y = b[2]; P = a[1] * b[1];
  ad2 mov MC0,X mov MUL,P mov MC1,Y mov ALU,A

  ; A += P; P = a[2] * b[2];
  ad2 mov MUL,P mov ALU,A
  
  ; A += P; PL = ALH
  ad2 mov ALU,A mov ALH,MC2

  ; Back to the loop
  btm
  nop

  ;=========================================================================
  ; Transfer up to CT0[36] dots to CT0[34]
  mov 34,CT0 clr A
  mov M0,A
  mov ALL,WA0

  mov 36,CT0
  mov M0,P

  ; CT[34] += CT[16]
  mov 34,CT0
  add mov ALL,MC0

  ; Transfer out and reset CT0[36]
  mov 36,CT0
  mov 0,CT2
  dma M2,D0,M0

  ; CT0[36] = 0; CT0[37] = 0;
  mov 36,CT0
  mvi 0,MC0
  mvi 0,MC0

WAIT_NOTRANSFER: jmp T0,WAIT_NOTRANSFER
  nop

  ; if (CT0[35]) goto COPYNORMALS
  mov 35,CT0 clr A
  mov M0,A
  mvi 0,PL
  add
  
  jmp NZ,COPYNORMALS
  nop

  endi

