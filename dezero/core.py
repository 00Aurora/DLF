import weakref
import numpy as np
import contextlib


class Config:
    enable_backprop = True # 역전파 활성

#--------------------------------------------------------------
@contextlib.contextmanager  #enable_backprop 온오프 설정
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():  # 
    return using_config('enable_backprop', False)

#--------------------------------------------------------------
class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        #  인수로 주어진 data가 None도 아니고 ndarray 인스턴스도 아니면
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self): #차원수
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self): #print
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def setup_variable():
        Variable.__add__ = add
        Variable.__radd__ = add
        Variable.__mul__ = mul
        Variable.__rmul__ = mul
        Variable.__neg__ = neg
        Variable.__sub__ = sub
        Variable.__rsub__ = rsub
        Variable.__truediv__ = div
        Variable.__rtruediv__ = rdiv
        Variable.__pow__ = pow


#--------------------------------------------------------------
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # 부모함수의 세대보다 1 큰 값

    def cleargrad(self):
        self.grad = None



        
#--------------------------------------------------------------
    def backward(self, retain_grad=False,create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data)) # 맨 첨 y.grad는 1의 array이므로

        funcs = [] #[self.creator]
        seen_set = set()
        def add_func(f): # 세대 순으로 정렬
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # () 약한 참조
            
            with using_config('enable_backprop',create_graph):
                gxs = f.backward(*gys)  #인수는 리스트 언팩, 메인 역전파
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx # 역전파로전파되는미분값을 변수 grad에 저장
                    else:   #  같은 변수 반복시
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            

            if not retain_grad:     # 말단 변수 외 미분값 안유지
                for y in f.outputs:
                    y().grad = None  # y is weakref

#--------------------------------------------------------------
def as_variable(obj):   #원소를 Variable로
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

# Variable을 항상 ndarray로
def as_array(x):        
    if np.isscalar(x):
        return np.array(x)
    return x

#--------------------------------------------------------------
class Function:
    def __call__(self, *inputs):# *별표로 붙인 인수 하나로 모아 받기
        inputs = [as_variable(x) for x in inputs] # 각각의 원소 x를 variable 인스턴스로 변환

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  #*은 리스트 언팩 : 언팩은리스트의원소를낱개로풀어서전달하는기법
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop: # 역전파 활성 모드라면
            self.generation = max([x.generation for x in inputs]) #변수가 여러개일 때 max
            for output in outputs:
                output.set_creator(self)    # 자신을 창조자로 세팅
            self.inputs = inputs # 참조카운트 1인분, 역전파시 함수에 입력한 변수가 필요시 사용
            self.outputs = [weakref.ref(output) for output in outputs] #약한 참조

        return outputs if len(outputs) > 1 else outputs[0]  # 리스

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

#--------------------------------------------------------------
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1 #순전파는 입력이 2개, 출력이 1개
        return y

    def backward(self, gy): #역전파는 반대
        return gy, gy #상류에서 흘러오는 미분값을 그대로 흘러보냄

def add(x0, x1):
    x1 = as_array(x1)   # Variable을 항상 ndarray로
    return Add()(x0, x1)


class Square(Function):
    def forward(self,x):
        return x ** 2

    def backward(self,gy):
        x = self.input[0].data
        gx = 2*x* gy
        return gx


class Mul(Function):
    def forward(self, x0, x1):#왼쪽 인수가 self, 오른쪽이 
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0



def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)



def setup_variable():
        Variable.__add__ = add
        Variable.__radd__ = add
        Variable.__mul__ = mul
        Variable.__rmul__ = mul
        Variable.__neg__ = neg
        Variable.__sub__ = sub
        Variable.__rsub__ = rsub
        Variable.__truediv__ = div
        Variable.__rtruediv__ = rdiv
        Variable.__pow__ = pow