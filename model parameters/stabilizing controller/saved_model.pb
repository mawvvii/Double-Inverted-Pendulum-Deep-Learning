??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
z
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z* 
shared_namehidden_1/kernel
s
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel*
_output_shapes

:Z*
dtype0
r
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namehidden_1/bias
k
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes
:Z*
dtype0
z
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ* 
shared_namehidden_2/kernel
s
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel*
_output_shapes

:ZZ*
dtype0
r
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namehidden_2/bias
k
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes
:Z*
dtype0
z
hidden_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ* 
shared_namehidden_3/kernel
s
#hidden_3/kernel/Read/ReadVariableOpReadVariableOphidden_3/kernel*
_output_shapes

:ZZ*
dtype0
r
hidden_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namehidden_3/bias
k
!hidden_3/bias/Read/ReadVariableOpReadVariableOphidden_3/bias*
_output_shapes
:Z*
dtype0
z
hidden_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ* 
shared_namehidden_4/kernel
s
#hidden_4/kernel/Read/ReadVariableOpReadVariableOphidden_4/kernel*
_output_shapes

:ZZ*
dtype0
r
hidden_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*
shared_namehidden_4/bias
k
!hidden_4/bias/Read/ReadVariableOpReadVariableOphidden_4/bias*
_output_shapes
:Z*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:Z*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/hidden_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*'
shared_nameAdam/hidden_1/kernel/m
?
*Adam/hidden_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/m*
_output_shapes

:Z*
dtype0
?
Adam/hidden_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_1/bias/m
y
(Adam/hidden_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/m*
_output_shapes
:Z*
dtype0
?
Adam/hidden_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*'
shared_nameAdam/hidden_2/kernel/m
?
*Adam/hidden_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/m*
_output_shapes

:ZZ*
dtype0
?
Adam/hidden_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_2/bias/m
y
(Adam/hidden_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/m*
_output_shapes
:Z*
dtype0
?
Adam/hidden_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*'
shared_nameAdam/hidden_3/kernel/m
?
*Adam/hidden_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_3/kernel/m*
_output_shapes

:ZZ*
dtype0
?
Adam/hidden_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_3/bias/m
y
(Adam/hidden_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_3/bias/m*
_output_shapes
:Z*
dtype0
?
Adam/hidden_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*'
shared_nameAdam/hidden_4/kernel/m
?
*Adam/hidden_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_4/kernel/m*
_output_shapes

:ZZ*
dtype0
?
Adam/hidden_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_4/bias/m
y
(Adam/hidden_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_4/bias/m*
_output_shapes
:Z*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:Z*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
?
Adam/hidden_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*'
shared_nameAdam/hidden_1/kernel/v
?
*Adam/hidden_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/v*
_output_shapes

:Z*
dtype0
?
Adam/hidden_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_1/bias/v
y
(Adam/hidden_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/v*
_output_shapes
:Z*
dtype0
?
Adam/hidden_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*'
shared_nameAdam/hidden_2/kernel/v
?
*Adam/hidden_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/v*
_output_shapes

:ZZ*
dtype0
?
Adam/hidden_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_2/bias/v
y
(Adam/hidden_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/v*
_output_shapes
:Z*
dtype0
?
Adam/hidden_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*'
shared_nameAdam/hidden_3/kernel/v
?
*Adam/hidden_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_3/kernel/v*
_output_shapes

:ZZ*
dtype0
?
Adam/hidden_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_3/bias/v
y
(Adam/hidden_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_3/bias/v*
_output_shapes
:Z*
dtype0
?
Adam/hidden_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ZZ*'
shared_nameAdam/hidden_4/kernel/v
?
*Adam/hidden_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_4/kernel/v*
_output_shapes

:ZZ*
dtype0
?
Adam/hidden_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z*%
shared_nameAdam/hidden_4/bias/v
y
(Adam/hidden_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_4/bias/v*
_output_shapes
:Z*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Z*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:Z*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?3
value?3B?3 B?3
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
?
*iter

+beta_1

,beta_2
	-decay
.learning_ratemRmSmTmUmVmWmXmY$mZ%m[v\v]v^v_v`vavbvc$vd%ve
 
F
0
1
2
3
4
5
6
7
$8
%9
F
0
1
2
3
4
5
6
7
$8
%9
?
regularization_losses
trainable_variables
/non_trainable_variables

0layers
1layer_regularization_losses
2layer_metrics
3metrics
		variables
 
[Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
trainable_variables
4non_trainable_variables
5layer_regularization_losses
6layer_metrics
7metrics

8layers
	variables
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
trainable_variables
9non_trainable_variables
:layer_regularization_losses
;layer_metrics
<metrics

=layers
	variables
[Y
VARIABLE_VALUEhidden_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
trainable_variables
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
Ametrics

Blayers
	variables
[Y
VARIABLE_VALUEhidden_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
 regularization_losses
!trainable_variables
Cnon_trainable_variables
Dlayer_regularization_losses
Elayer_metrics
Fmetrics

Glayers
"	variables
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
?
&regularization_losses
'trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses
Jlayer_metrics
Kmetrics

Llayers
(	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 
 

M0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ntotal
	Ocount
P	variables
Q	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

N0
O1

P	variables
~|
VARIABLE_VALUEAdam/hidden_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_hidden_1_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_hidden_1_inputhidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biashidden_4/kernelhidden_4/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_30162947
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp#hidden_3/kernel/Read/ReadVariableOp!hidden_3/bias/Read/ReadVariableOp#hidden_4/kernel/Read/ReadVariableOp!hidden_4/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/hidden_1/kernel/m/Read/ReadVariableOp(Adam/hidden_1/bias/m/Read/ReadVariableOp*Adam/hidden_2/kernel/m/Read/ReadVariableOp(Adam/hidden_2/bias/m/Read/ReadVariableOp*Adam/hidden_3/kernel/m/Read/ReadVariableOp(Adam/hidden_3/bias/m/Read/ReadVariableOp*Adam/hidden_4/kernel/m/Read/ReadVariableOp(Adam/hidden_4/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp*Adam/hidden_1/kernel/v/Read/ReadVariableOp(Adam/hidden_1/bias/v/Read/ReadVariableOp*Adam/hidden_2/kernel/v/Read/ReadVariableOp(Adam/hidden_2/bias/v/Read/ReadVariableOp*Adam/hidden_3/kernel/v/Read/ReadVariableOp(Adam/hidden_3/bias/v/Read/ReadVariableOp*Adam/hidden_4/kernel/v/Read/ReadVariableOp(Adam/hidden_4/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_30163306
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biashidden_3/kernelhidden_3/biashidden_4/kernelhidden_4/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/hidden_1/kernel/mAdam/hidden_1/bias/mAdam/hidden_2/kernel/mAdam/hidden_2/bias/mAdam/hidden_3/kernel/mAdam/hidden_3/bias/mAdam/hidden_4/kernel/mAdam/hidden_4/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/hidden_1/kernel/vAdam/hidden_1/bias/vAdam/hidden_2/kernel/vAdam/hidden_2/bias/vAdam/hidden_3/kernel/vAdam/hidden_3/bias/vAdam/hidden_4/kernel/vAdam/hidden_4/bias/vAdam/output/kernel/vAdam/output/bias/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_30163427??
?<
?	
#__inference__wrapped_model_30162587
hidden_1_inputF
4simple_model_hidden_1_matmul_readvariableop_resource:ZC
5simple_model_hidden_1_biasadd_readvariableop_resource:ZF
4simple_model_hidden_2_matmul_readvariableop_resource:ZZC
5simple_model_hidden_2_biasadd_readvariableop_resource:ZF
4simple_model_hidden_3_matmul_readvariableop_resource:ZZC
5simple_model_hidden_3_biasadd_readvariableop_resource:ZF
4simple_model_hidden_4_matmul_readvariableop_resource:ZZC
5simple_model_hidden_4_biasadd_readvariableop_resource:ZD
2simple_model_output_matmul_readvariableop_resource:ZA
3simple_model_output_biasadd_readvariableop_resource:
identity??,simple_model/hidden_1/BiasAdd/ReadVariableOp?+simple_model/hidden_1/MatMul/ReadVariableOp?,simple_model/hidden_2/BiasAdd/ReadVariableOp?+simple_model/hidden_2/MatMul/ReadVariableOp?,simple_model/hidden_3/BiasAdd/ReadVariableOp?+simple_model/hidden_3/MatMul/ReadVariableOp?,simple_model/hidden_4/BiasAdd/ReadVariableOp?+simple_model/hidden_4/MatMul/ReadVariableOp?*simple_model/output/BiasAdd/ReadVariableOp?)simple_model/output/MatMul/ReadVariableOp?
+simple_model/hidden_1/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02-
+simple_model/hidden_1/MatMul/ReadVariableOp?
simple_model/hidden_1/MatMulMatMulhidden_1_input3simple_model/hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_1/MatMul?
,simple_model/hidden_1/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_1_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02.
,simple_model/hidden_1/BiasAdd/ReadVariableOp?
simple_model/hidden_1/BiasAddBiasAdd&simple_model/hidden_1/MatMul:product:04simple_model/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_1/BiasAdd?
simple_model/hidden_1/TanhTanh&simple_model/hidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_1/Tanh?
+simple_model/hidden_2/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_2_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02-
+simple_model/hidden_2/MatMul/ReadVariableOp?
simple_model/hidden_2/MatMulMatMulsimple_model/hidden_1/Tanh:y:03simple_model/hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_2/MatMul?
,simple_model/hidden_2/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_2_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02.
,simple_model/hidden_2/BiasAdd/ReadVariableOp?
simple_model/hidden_2/BiasAddBiasAdd&simple_model/hidden_2/MatMul:product:04simple_model/hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_2/BiasAdd?
simple_model/hidden_2/TanhTanh&simple_model/hidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_2/Tanh?
+simple_model/hidden_3/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_3_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02-
+simple_model/hidden_3/MatMul/ReadVariableOp?
simple_model/hidden_3/MatMulMatMulsimple_model/hidden_2/Tanh:y:03simple_model/hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_3/MatMul?
,simple_model/hidden_3/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02.
,simple_model/hidden_3/BiasAdd/ReadVariableOp?
simple_model/hidden_3/BiasAddBiasAdd&simple_model/hidden_3/MatMul:product:04simple_model/hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_3/BiasAdd?
simple_model/hidden_3/TanhTanh&simple_model/hidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_3/Tanh?
+simple_model/hidden_4/MatMul/ReadVariableOpReadVariableOp4simple_model_hidden_4_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02-
+simple_model/hidden_4/MatMul/ReadVariableOp?
simple_model/hidden_4/MatMulMatMulsimple_model/hidden_3/Tanh:y:03simple_model/hidden_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_4/MatMul?
,simple_model/hidden_4/BiasAdd/ReadVariableOpReadVariableOp5simple_model_hidden_4_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02.
,simple_model/hidden_4/BiasAdd/ReadVariableOp?
simple_model/hidden_4/BiasAddBiasAdd&simple_model/hidden_4/MatMul:product:04simple_model/hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_4/BiasAdd?
simple_model/hidden_4/TanhTanh&simple_model/hidden_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
simple_model/hidden_4/Tanh?
)simple_model/output/MatMul/ReadVariableOpReadVariableOp2simple_model_output_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02+
)simple_model/output/MatMul/ReadVariableOp?
simple_model/output/MatMulMatMulsimple_model/hidden_4/Tanh:y:01simple_model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_model/output/MatMul?
*simple_model/output/BiasAdd/ReadVariableOpReadVariableOp3simple_model_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*simple_model/output/BiasAdd/ReadVariableOp?
simple_model/output/BiasAddBiasAdd$simple_model/output/MatMul:product:02simple_model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
simple_model/output/BiasAdd?
IdentityIdentity$simple_model/output/BiasAdd:output:0-^simple_model/hidden_1/BiasAdd/ReadVariableOp,^simple_model/hidden_1/MatMul/ReadVariableOp-^simple_model/hidden_2/BiasAdd/ReadVariableOp,^simple_model/hidden_2/MatMul/ReadVariableOp-^simple_model/hidden_3/BiasAdd/ReadVariableOp,^simple_model/hidden_3/MatMul/ReadVariableOp-^simple_model/hidden_4/BiasAdd/ReadVariableOp,^simple_model/hidden_4/MatMul/ReadVariableOp+^simple_model/output/BiasAdd/ReadVariableOp*^simple_model/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2\
,simple_model/hidden_1/BiasAdd/ReadVariableOp,simple_model/hidden_1/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_1/MatMul/ReadVariableOp+simple_model/hidden_1/MatMul/ReadVariableOp2\
,simple_model/hidden_2/BiasAdd/ReadVariableOp,simple_model/hidden_2/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_2/MatMul/ReadVariableOp+simple_model/hidden_2/MatMul/ReadVariableOp2\
,simple_model/hidden_3/BiasAdd/ReadVariableOp,simple_model/hidden_3/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_3/MatMul/ReadVariableOp+simple_model/hidden_3/MatMul/ReadVariableOp2\
,simple_model/hidden_4/BiasAdd/ReadVariableOp,simple_model/hidden_4/BiasAdd/ReadVariableOp2Z
+simple_model/hidden_4/MatMul/ReadVariableOp+simple_model/hidden_4/MatMul/ReadVariableOp2X
*simple_model/output/BiasAdd/ReadVariableOp*simple_model/output/BiasAdd/ReadVariableOp2V
)simple_model/output/MatMul/ReadVariableOp)simple_model/output/MatMul/ReadVariableOp:W S
'
_output_shapes
:?????????
(
_user_specified_namehidden_1_input
?

?
F__inference_hidden_1_layer_call_and_return_conditional_losses_30162605

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_output_layer_call_and_return_conditional_losses_30162672

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_30163427
file_prefix2
 assignvariableop_hidden_1_kernel:Z.
 assignvariableop_1_hidden_1_bias:Z4
"assignvariableop_2_hidden_2_kernel:ZZ.
 assignvariableop_3_hidden_2_bias:Z4
"assignvariableop_4_hidden_3_kernel:ZZ.
 assignvariableop_5_hidden_3_bias:Z4
"assignvariableop_6_hidden_4_kernel:ZZ.
 assignvariableop_7_hidden_4_bias:Z2
 assignvariableop_8_output_kernel:Z,
assignvariableop_9_output_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: <
*assignvariableop_17_adam_hidden_1_kernel_m:Z6
(assignvariableop_18_adam_hidden_1_bias_m:Z<
*assignvariableop_19_adam_hidden_2_kernel_m:ZZ6
(assignvariableop_20_adam_hidden_2_bias_m:Z<
*assignvariableop_21_adam_hidden_3_kernel_m:ZZ6
(assignvariableop_22_adam_hidden_3_bias_m:Z<
*assignvariableop_23_adam_hidden_4_kernel_m:ZZ6
(assignvariableop_24_adam_hidden_4_bias_m:Z:
(assignvariableop_25_adam_output_kernel_m:Z4
&assignvariableop_26_adam_output_bias_m:<
*assignvariableop_27_adam_hidden_1_kernel_v:Z6
(assignvariableop_28_adam_hidden_1_bias_v:Z<
*assignvariableop_29_adam_hidden_2_kernel_v:ZZ6
(assignvariableop_30_adam_hidden_2_bias_v:Z<
*assignvariableop_31_adam_hidden_3_kernel_v:ZZ6
(assignvariableop_32_adam_hidden_3_bias_v:Z<
*assignvariableop_33_adam_hidden_4_kernel_v:ZZ6
(assignvariableop_34_adam_hidden_4_bias_v:Z:
(assignvariableop_35_adam_output_kernel_v:Z4
&assignvariableop_36_adam_output_bias_v:
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_hidden_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_hidden_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_hidden_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_hidden_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_hidden_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_hidden_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_hidden_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_hidden_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_hidden_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_hidden_3_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_hidden_3_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_hidden_4_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_hidden_4_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_output_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_output_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_hidden_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_hidden_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_hidden_2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_hidden_2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_hidden_3_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_hidden_3_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_hidden_4_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_hidden_4_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_output_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_output_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
F__inference_hidden_4_layer_call_and_return_conditional_losses_30163153

inputs0
matmul_readvariableop_resource:ZZ-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
?
+__inference_hidden_3_layer_call_fn_30163122

inputs
unknown:ZZ
	unknown_0:Z
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_3_layer_call_and_return_conditional_losses_301626392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?

?
F__inference_hidden_3_layer_call_and_return_conditional_losses_30162639

inputs0
matmul_readvariableop_resource:ZZ-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
?
)__inference_output_layer_call_fn_30163162

inputs
unknown:Z
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_301626722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
?
J__inference_simple_model_layer_call_and_return_conditional_losses_30162885
hidden_1_input#
hidden_1_30162859:Z
hidden_1_30162861:Z#
hidden_2_30162864:ZZ
hidden_2_30162866:Z#
hidden_3_30162869:ZZ
hidden_3_30162871:Z#
hidden_4_30162874:ZZ
hidden_4_30162876:Z!
output_30162879:Z
output_30162881:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputhidden_1_30162859hidden_1_30162861*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_1_layer_call_and_return_conditional_losses_301626052"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_30162864hidden_2_30162866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_2_layer_call_and_return_conditional_losses_301626222"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_30162869hidden_3_30162871*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_3_layer_call_and_return_conditional_losses_301626392"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_30162874hidden_4_30162876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_4_layer_call_and_return_conditional_losses_301626562"
 hidden_4/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_30162879output_30162881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_301626722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namehidden_1_input
?

?
F__inference_hidden_4_layer_call_and_return_conditional_losses_30162656

inputs0
matmul_readvariableop_resource:ZZ-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?/
?
J__inference_simple_model_layer_call_and_return_conditional_losses_30163035

inputs9
'hidden_1_matmul_readvariableop_resource:Z6
(hidden_1_biasadd_readvariableop_resource:Z9
'hidden_2_matmul_readvariableop_resource:ZZ6
(hidden_2_biasadd_readvariableop_resource:Z9
'hidden_3_matmul_readvariableop_resource:ZZ6
(hidden_3_biasadd_readvariableop_resource:Z9
'hidden_4_matmul_readvariableop_resource:ZZ6
(hidden_4_biasadd_readvariableop_resource:Z7
%output_matmul_readvariableop_resource:Z4
&output_biasadd_readvariableop_resource:
identity??hidden_1/BiasAdd/ReadVariableOp?hidden_1/MatMul/ReadVariableOp?hidden_2/BiasAdd/ReadVariableOp?hidden_2/MatMul/ReadVariableOp?hidden_3/BiasAdd/ReadVariableOp?hidden_3/MatMul/ReadVariableOp?hidden_4/BiasAdd/ReadVariableOp?hidden_4/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02 
hidden_1/MatMul/ReadVariableOp?
hidden_1/MatMulMatMulinputs&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_1/MatMul?
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_1/BiasAdd/ReadVariableOp?
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_1/BiasAdds
hidden_1/TanhTanhhidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_1/Tanh?
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02 
hidden_2/MatMul/ReadVariableOp?
hidden_2/MatMulMatMulhidden_1/Tanh:y:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_2/MatMul?
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_2/BiasAdd/ReadVariableOp?
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_2/BiasAdds
hidden_2/TanhTanhhidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_2/Tanh?
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02 
hidden_3/MatMul/ReadVariableOp?
hidden_3/MatMulMatMulhidden_2/Tanh:y:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_3/MatMul?
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_3/BiasAdd/ReadVariableOp?
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_3/BiasAdds
hidden_3/TanhTanhhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_3/Tanh?
hidden_4/MatMul/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02 
hidden_4/MatMul/ReadVariableOp?
hidden_4/MatMulMatMulhidden_3/Tanh:y:0&hidden_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_4/MatMul?
hidden_4/BiasAdd/ReadVariableOpReadVariableOp(hidden_4_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_4/BiasAdd/ReadVariableOp?
hidden_4/BiasAddBiasAddhidden_4/MatMul:product:0'hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_4/BiasAdds
hidden_4/TanhTanhhidden_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_4/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulhidden_4/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0 ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp ^hidden_4/BiasAdd/ReadVariableOp^hidden_4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2B
hidden_4/BiasAdd/ReadVariableOphidden_4/BiasAdd/ReadVariableOp2@
hidden_4/MatMul/ReadVariableOphidden_4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
J__inference_simple_model_layer_call_and_return_conditional_losses_30163073

inputs9
'hidden_1_matmul_readvariableop_resource:Z6
(hidden_1_biasadd_readvariableop_resource:Z9
'hidden_2_matmul_readvariableop_resource:ZZ6
(hidden_2_biasadd_readvariableop_resource:Z9
'hidden_3_matmul_readvariableop_resource:ZZ6
(hidden_3_biasadd_readvariableop_resource:Z9
'hidden_4_matmul_readvariableop_resource:ZZ6
(hidden_4_biasadd_readvariableop_resource:Z7
%output_matmul_readvariableop_resource:Z4
&output_biasadd_readvariableop_resource:
identity??hidden_1/BiasAdd/ReadVariableOp?hidden_1/MatMul/ReadVariableOp?hidden_2/BiasAdd/ReadVariableOp?hidden_2/MatMul/ReadVariableOp?hidden_3/BiasAdd/ReadVariableOp?hidden_3/MatMul/ReadVariableOp?hidden_4/BiasAdd/ReadVariableOp?hidden_4/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02 
hidden_1/MatMul/ReadVariableOp?
hidden_1/MatMulMatMulinputs&hidden_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_1/MatMul?
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_1/BiasAdd/ReadVariableOp?
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_1/BiasAdds
hidden_1/TanhTanhhidden_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_1/Tanh?
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02 
hidden_2/MatMul/ReadVariableOp?
hidden_2/MatMulMatMulhidden_1/Tanh:y:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_2/MatMul?
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_2/BiasAdd/ReadVariableOp?
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_2/BiasAdds
hidden_2/TanhTanhhidden_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_2/Tanh?
hidden_3/MatMul/ReadVariableOpReadVariableOp'hidden_3_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02 
hidden_3/MatMul/ReadVariableOp?
hidden_3/MatMulMatMulhidden_2/Tanh:y:0&hidden_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_3/MatMul?
hidden_3/BiasAdd/ReadVariableOpReadVariableOp(hidden_3_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_3/BiasAdd/ReadVariableOp?
hidden_3/BiasAddBiasAddhidden_3/MatMul:product:0'hidden_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_3/BiasAdds
hidden_3/TanhTanhhidden_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_3/Tanh?
hidden_4/MatMul/ReadVariableOpReadVariableOp'hidden_4_matmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02 
hidden_4/MatMul/ReadVariableOp?
hidden_4/MatMulMatMulhidden_3/Tanh:y:0&hidden_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_4/MatMul?
hidden_4/BiasAdd/ReadVariableOpReadVariableOp(hidden_4_biasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02!
hidden_4/BiasAdd/ReadVariableOp?
hidden_4/BiasAddBiasAddhidden_4/MatMul:product:0'hidden_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
hidden_4/BiasAdds
hidden_4/TanhTanhhidden_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
hidden_4/Tanh?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMulhidden_4/Tanh:y:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0 ^hidden_1/BiasAdd/ReadVariableOp^hidden_1/MatMul/ReadVariableOp ^hidden_2/BiasAdd/ReadVariableOp^hidden_2/MatMul/ReadVariableOp ^hidden_3/BiasAdd/ReadVariableOp^hidden_3/MatMul/ReadVariableOp ^hidden_4/BiasAdd/ReadVariableOp^hidden_4/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2B
hidden_1/BiasAdd/ReadVariableOphidden_1/BiasAdd/ReadVariableOp2@
hidden_1/MatMul/ReadVariableOphidden_1/MatMul/ReadVariableOp2B
hidden_2/BiasAdd/ReadVariableOphidden_2/BiasAdd/ReadVariableOp2@
hidden_2/MatMul/ReadVariableOphidden_2/MatMul/ReadVariableOp2B
hidden_3/BiasAdd/ReadVariableOphidden_3/BiasAdd/ReadVariableOp2@
hidden_3/MatMul/ReadVariableOphidden_3/MatMul/ReadVariableOp2B
hidden_4/BiasAdd/ReadVariableOphidden_4/BiasAdd/ReadVariableOp2@
hidden_4/MatMul/ReadVariableOphidden_4/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_simple_model_layer_call_and_return_conditional_losses_30162808

inputs#
hidden_1_30162782:Z
hidden_1_30162784:Z#
hidden_2_30162787:ZZ
hidden_2_30162789:Z#
hidden_3_30162792:ZZ
hidden_3_30162794:Z#
hidden_4_30162797:ZZ
hidden_4_30162799:Z!
output_30162802:Z
output_30162804:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_30162782hidden_1_30162784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_1_layer_call_and_return_conditional_losses_301626052"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_30162787hidden_2_30162789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_2_layer_call_and_return_conditional_losses_301626222"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_30162792hidden_3_30162794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_3_layer_call_and_return_conditional_losses_301626392"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_30162797hidden_4_30162799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_4_layer_call_and_return_conditional_losses_301626562"
 hidden_4/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_30162802output_30162804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_301626722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_simple_model_layer_call_fn_30162972

inputs
unknown:Z
	unknown_0:Z
	unknown_1:ZZ
	unknown_2:Z
	unknown_3:ZZ
	unknown_4:Z
	unknown_5:ZZ
	unknown_6:Z
	unknown_7:Z
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_model_layer_call_and_return_conditional_losses_301626792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
F__inference_hidden_3_layer_call_and_return_conditional_losses_30163133

inputs0
matmul_readvariableop_resource:ZZ-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?	
?
/__inference_simple_model_layer_call_fn_30162997

inputs
unknown:Z
	unknown_0:Z
	unknown_1:ZZ
	unknown_2:Z
	unknown_3:ZZ
	unknown_4:Z
	unknown_5:ZZ
	unknown_6:Z
	unknown_7:Z
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_model_layer_call_and_return_conditional_losses_301628082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_hidden_4_layer_call_fn_30163142

inputs
unknown:ZZ
	unknown_0:Z
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_4_layer_call_and_return_conditional_losses_301626562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_30162947
hidden_1_input
unknown:Z
	unknown_0:Z
	unknown_1:ZZ
	unknown_2:Z
	unknown_3:ZZ
	unknown_4:Z
	unknown_5:ZZ
	unknown_6:Z
	unknown_7:Z
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_301625872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namehidden_1_input
?

?
/__inference_simple_model_layer_call_fn_30162702
hidden_1_input
unknown:Z
	unknown_0:Z
	unknown_1:ZZ
	unknown_2:Z
	unknown_3:ZZ
	unknown_4:Z
	unknown_5:ZZ
	unknown_6:Z
	unknown_7:Z
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_model_layer_call_and_return_conditional_losses_301626792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namehidden_1_input
?
?
+__inference_hidden_2_layer_call_fn_30163102

inputs
unknown:ZZ
	unknown_0:Z
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_2_layer_call_and_return_conditional_losses_301626222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
?
+__inference_hidden_1_layer_call_fn_30163082

inputs
unknown:Z
	unknown_0:Z
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_1_layer_call_and_return_conditional_losses_301626052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
/__inference_simple_model_layer_call_fn_30162856
hidden_1_input
unknown:Z
	unknown_0:Z
	unknown_1:ZZ
	unknown_2:Z
	unknown_3:ZZ
	unknown_4:Z
	unknown_5:ZZ
	unknown_6:Z
	unknown_7:Z
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_model_layer_call_and_return_conditional_losses_301628082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namehidden_1_input
?

?
F__inference_hidden_2_layer_call_and_return_conditional_losses_30163113

inputs0
matmul_readvariableop_resource:ZZ-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?

?
F__inference_hidden_2_layer_call_and_return_conditional_losses_30162622

inputs0
matmul_readvariableop_resource:ZZ-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ZZ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?
?
J__inference_simple_model_layer_call_and_return_conditional_losses_30162914
hidden_1_input#
hidden_1_30162888:Z
hidden_1_30162890:Z#
hidden_2_30162893:ZZ
hidden_2_30162895:Z#
hidden_3_30162898:ZZ
hidden_3_30162900:Z#
hidden_4_30162903:ZZ
hidden_4_30162905:Z!
output_30162908:Z
output_30162910:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputhidden_1_30162888hidden_1_30162890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_1_layer_call_and_return_conditional_losses_301626052"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_30162893hidden_2_30162895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_2_layer_call_and_return_conditional_losses_301626222"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_30162898hidden_3_30162900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_3_layer_call_and_return_conditional_losses_301626392"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_30162903hidden_4_30162905*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_4_layer_call_and_return_conditional_losses_301626562"
 hidden_4/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_30162908output_30162910*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_301626722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namehidden_1_input
?
?
J__inference_simple_model_layer_call_and_return_conditional_losses_30162679

inputs#
hidden_1_30162606:Z
hidden_1_30162608:Z#
hidden_2_30162623:ZZ
hidden_2_30162625:Z#
hidden_3_30162640:ZZ
hidden_3_30162642:Z#
hidden_4_30162657:ZZ
hidden_4_30162659:Z!
output_30162673:Z
output_30162675:
identity?? hidden_1/StatefulPartitionedCall? hidden_2/StatefulPartitionedCall? hidden_3/StatefulPartitionedCall? hidden_4/StatefulPartitionedCall?output/StatefulPartitionedCall?
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_30162606hidden_1_30162608*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_1_layer_call_and_return_conditional_losses_301626052"
 hidden_1/StatefulPartitionedCall?
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0hidden_2_30162623hidden_2_30162625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_2_layer_call_and_return_conditional_losses_301626222"
 hidden_2/StatefulPartitionedCall?
 hidden_3/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0hidden_3_30162640hidden_3_30162642*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_3_layer_call_and_return_conditional_losses_301626392"
 hidden_3/StatefulPartitionedCall?
 hidden_4/StatefulPartitionedCallStatefulPartitionedCall)hidden_3/StatefulPartitionedCall:output:0hidden_4_30162657hidden_4_30162659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????Z*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_hidden_4_layer_call_and_return_conditional_losses_301626562"
 hidden_4/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall)hidden_4/StatefulPartitionedCall:output:0output_30162673output_30162675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_output_layer_call_and_return_conditional_losses_301626722 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall!^hidden_3/StatefulPartitionedCall!^hidden_4/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2D
 hidden_3/StatefulPartitionedCall hidden_3/StatefulPartitionedCall2D
 hidden_4/StatefulPartitionedCall hidden_4/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_output_layer_call_and_return_conditional_losses_30163172

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????Z
 
_user_specified_nameinputs
?O
?
!__inference__traced_save_30163306
file_prefix.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop.
*savev2_hidden_3_kernel_read_readvariableop,
(savev2_hidden_3_bias_read_readvariableop.
*savev2_hidden_4_kernel_read_readvariableop,
(savev2_hidden_4_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_hidden_1_kernel_m_read_readvariableop3
/savev2_adam_hidden_1_bias_m_read_readvariableop5
1savev2_adam_hidden_2_kernel_m_read_readvariableop3
/savev2_adam_hidden_2_bias_m_read_readvariableop5
1savev2_adam_hidden_3_kernel_m_read_readvariableop3
/savev2_adam_hidden_3_bias_m_read_readvariableop5
1savev2_adam_hidden_4_kernel_m_read_readvariableop3
/savev2_adam_hidden_4_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop5
1savev2_adam_hidden_1_kernel_v_read_readvariableop3
/savev2_adam_hidden_1_bias_v_read_readvariableop5
1savev2_adam_hidden_2_kernel_v_read_readvariableop3
/savev2_adam_hidden_2_bias_v_read_readvariableop5
1savev2_adam_hidden_3_kernel_v_read_readvariableop3
/savev2_adam_hidden_3_bias_v_read_readvariableop5
1savev2_adam_hidden_4_kernel_v_read_readvariableop3
/savev2_adam_hidden_4_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop*savev2_hidden_3_kernel_read_readvariableop(savev2_hidden_3_bias_read_readvariableop*savev2_hidden_4_kernel_read_readvariableop(savev2_hidden_4_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_hidden_1_kernel_m_read_readvariableop/savev2_adam_hidden_1_bias_m_read_readvariableop1savev2_adam_hidden_2_kernel_m_read_readvariableop/savev2_adam_hidden_2_bias_m_read_readvariableop1savev2_adam_hidden_3_kernel_m_read_readvariableop/savev2_adam_hidden_3_bias_m_read_readvariableop1savev2_adam_hidden_4_kernel_m_read_readvariableop/savev2_adam_hidden_4_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop1savev2_adam_hidden_1_kernel_v_read_readvariableop/savev2_adam_hidden_1_bias_v_read_readvariableop1savev2_adam_hidden_2_kernel_v_read_readvariableop/savev2_adam_hidden_2_bias_v_read_readvariableop1savev2_adam_hidden_3_kernel_v_read_readvariableop/savev2_adam_hidden_3_bias_v_read_readvariableop1savev2_adam_hidden_4_kernel_v_read_readvariableop/savev2_adam_hidden_4_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :Z:Z:ZZ:Z:ZZ:Z:ZZ:Z:Z:: : : : : : : :Z:Z:ZZ:Z:ZZ:Z:ZZ:Z:Z::Z:Z:ZZ:Z:ZZ:Z:ZZ:Z:Z:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:Z: 

_output_shapes
:Z:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z:$	 

_output_shapes

:Z: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:Z: 

_output_shapes
:Z:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z:$ 

_output_shapes

:Z: 

_output_shapes
::$ 

_output_shapes

:Z: 

_output_shapes
:Z:$ 

_output_shapes

:ZZ: 

_output_shapes
:Z:$  

_output_shapes

:ZZ: !

_output_shapes
:Z:$" 

_output_shapes

:ZZ: #

_output_shapes
:Z:$$ 

_output_shapes

:Z: %

_output_shapes
::&

_output_shapes
: 
?

?
F__inference_hidden_1_layer_call_and_return_conditional_losses_30163093

inputs0
matmul_readvariableop_resource:Z-
biasadd_readvariableop_resource:Z
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Z*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Z*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
hidden_1_input7
 serving_default_hidden_1_input:0?????????:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?4
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
f__call__
g_default_save_signature
*h&call_and_return_all_conditional_losses"?1
_tf_keras_sequential?1{"name": "simple_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "simple_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hidden_1_input"}}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "hidden_4", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 6]}, "float32", "hidden_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "simple_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hidden_1_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "hidden_4", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 4.999999873689376e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
i__call__
*j&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6]}, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_3", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
?

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
o__call__
*p&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "hidden_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "hidden_4", "trainable": true, "dtype": "float32", "units": 90, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
?

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
q__call__
*r&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 90}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 90]}}
?
*iter

+beta_1

,beta_2
	-decay
.learning_ratemRmSmTmUmVmWmXmY$mZ%m[v\v]v^v_v`vavbvc$vd%ve"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
?
regularization_losses
trainable_variables
/non_trainable_variables

0layers
1layer_regularization_losses
2layer_metrics
3metrics
		variables
f__call__
g_default_save_signature
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
,
sserving_default"
signature_map
!:Z2hidden_1/kernel
:Z2hidden_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
trainable_variables
4non_trainable_variables
5layer_regularization_losses
6layer_metrics
7metrics

8layers
	variables
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
!:ZZ2hidden_2/kernel
:Z2hidden_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
trainable_variables
9non_trainable_variables
:layer_regularization_losses
;layer_metrics
<metrics

=layers
	variables
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
!:ZZ2hidden_3/kernel
:Z2hidden_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
trainable_variables
>non_trainable_variables
?layer_regularization_losses
@layer_metrics
Ametrics

Blayers
	variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
!:ZZ2hidden_4/kernel
:Z2hidden_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
 regularization_losses
!trainable_variables
Cnon_trainable_variables
Dlayer_regularization_losses
Elayer_metrics
Fmetrics

Glayers
"	variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
:Z2output/kernel
:2output/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
&regularization_losses
'trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses
Jlayer_metrics
Kmetrics

Llayers
(	variables
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
M0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Ntotal
	Ocount
P	variables
Q	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 22}
:  (2total
:  (2count
.
N0
O1"
trackable_list_wrapper
-
P	variables"
_generic_user_object
&:$Z2Adam/hidden_1/kernel/m
 :Z2Adam/hidden_1/bias/m
&:$ZZ2Adam/hidden_2/kernel/m
 :Z2Adam/hidden_2/bias/m
&:$ZZ2Adam/hidden_3/kernel/m
 :Z2Adam/hidden_3/bias/m
&:$ZZ2Adam/hidden_4/kernel/m
 :Z2Adam/hidden_4/bias/m
$:"Z2Adam/output/kernel/m
:2Adam/output/bias/m
&:$Z2Adam/hidden_1/kernel/v
 :Z2Adam/hidden_1/bias/v
&:$ZZ2Adam/hidden_2/kernel/v
 :Z2Adam/hidden_2/bias/v
&:$ZZ2Adam/hidden_3/kernel/v
 :Z2Adam/hidden_3/bias/v
&:$ZZ2Adam/hidden_4/kernel/v
 :Z2Adam/hidden_4/bias/v
$:"Z2Adam/output/kernel/v
:2Adam/output/bias/v
?2?
/__inference_simple_model_layer_call_fn_30162702
/__inference_simple_model_layer_call_fn_30162972
/__inference_simple_model_layer_call_fn_30162997
/__inference_simple_model_layer_call_fn_30162856?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_30162587?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *-?*
(?%
hidden_1_input?????????
?2?
J__inference_simple_model_layer_call_and_return_conditional_losses_30163035
J__inference_simple_model_layer_call_and_return_conditional_losses_30163073
J__inference_simple_model_layer_call_and_return_conditional_losses_30162885
J__inference_simple_model_layer_call_and_return_conditional_losses_30162914?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_hidden_1_layer_call_fn_30163082?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_hidden_1_layer_call_and_return_conditional_losses_30163093?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_hidden_2_layer_call_fn_30163102?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_hidden_2_layer_call_and_return_conditional_losses_30163113?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_hidden_3_layer_call_fn_30163122?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_hidden_3_layer_call_and_return_conditional_losses_30163133?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_hidden_4_layer_call_fn_30163142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_hidden_4_layer_call_and_return_conditional_losses_30163153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_output_layer_call_fn_30163162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_output_layer_call_and_return_conditional_losses_30163172?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_30162947hidden_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_30162587v
$%7?4
-?*
(?%
hidden_1_input?????????
? "/?,
*
output ?
output??????????
F__inference_hidden_1_layer_call_and_return_conditional_losses_30163093\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????Z
? ~
+__inference_hidden_1_layer_call_fn_30163082O/?,
%?"
 ?
inputs?????????
? "??????????Z?
F__inference_hidden_2_layer_call_and_return_conditional_losses_30163113\/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????Z
? ~
+__inference_hidden_2_layer_call_fn_30163102O/?,
%?"
 ?
inputs?????????Z
? "??????????Z?
F__inference_hidden_3_layer_call_and_return_conditional_losses_30163133\/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????Z
? ~
+__inference_hidden_3_layer_call_fn_30163122O/?,
%?"
 ?
inputs?????????Z
? "??????????Z?
F__inference_hidden_4_layer_call_and_return_conditional_losses_30163153\/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????Z
? ~
+__inference_hidden_4_layer_call_fn_30163142O/?,
%?"
 ?
inputs?????????Z
? "??????????Z?
D__inference_output_layer_call_and_return_conditional_losses_30163172\$%/?,
%?"
 ?
inputs?????????Z
? "%?"
?
0?????????
? |
)__inference_output_layer_call_fn_30163162O$%/?,
%?"
 ?
inputs?????????Z
? "???????????
&__inference_signature_wrapper_30162947?
$%I?F
? 
??<
:
hidden_1_input(?%
hidden_1_input?????????"/?,
*
output ?
output??????????
J__inference_simple_model_layer_call_and_return_conditional_losses_30162885t
$%??<
5?2
(?%
hidden_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_simple_model_layer_call_and_return_conditional_losses_30162914t
$%??<
5?2
(?%
hidden_1_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_simple_model_layer_call_and_return_conditional_losses_30163035l
$%7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_simple_model_layer_call_and_return_conditional_losses_30163073l
$%7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_simple_model_layer_call_fn_30162702g
$%??<
5?2
(?%
hidden_1_input?????????
p 

 
? "???????????
/__inference_simple_model_layer_call_fn_30162856g
$%??<
5?2
(?%
hidden_1_input?????????
p

 
? "???????????
/__inference_simple_model_layer_call_fn_30162972_
$%7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
/__inference_simple_model_layer_call_fn_30162997_
$%7?4
-?*
 ?
inputs?????????
p

 
? "??????????