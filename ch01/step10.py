from typing import Any
import numpy as np

# 태스트셋 라이브러리
import unittest

class Variable:
	def __init__(self,data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{}은 지원하지 않습니다'.format(type(data)))

		self.data = data # 통상값
		self.grad = None # 미분값
		self.creator = None # 변수 관점에서 함수는 creator(부모)
	
	def set_creator(self, func):
		self.creator = func
	
	# loop
	def backward(self):
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = [self.creator]
		while funcs:
			f = funcs.pop() # 함수를 가져옴
			x,y = f.input, f.output # 함수의 입력과 출력을 가져옴
			x.grad = f.backward(y.grad) # backward 매서드 호출

			if x.creator is not None:
				funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가
	

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(as_array(y))
		output.set_creator(self) # 출력 변수에 창조자를 설정
		self.input = input # 입력 변수를 보관(기억)
		self.output = output # 출력도 저장(기억)
		return output
	
	def forward(self, x):
		raise NotImplementedError

	# 미분을 계산하는 역전파
	def backward(self, gy):
		raise NotImplementedError


class Square(Function):
	# 제곱 연산
	def forward(self, x):
		y = x**2
		return y
	
	# 제곱 미분
	def backward(self, gy):
		x = self.input.data
		gx = 2*x*gy
		return gx
	
class Exp(Function):
	def forward(self, x):
		y = np.exp(x)
		return y
	
	def backward(self, gy):
		x = self.input.data
		gx = np.exp(x) * gy
		return gx
	
def square(x):
	f = Square()
	return f(as_array(x))

def exp(x):
	f = Exp()
	return f(x)

# ndarray 보장 해주는 함수
def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x

def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(as_array(x.data - eps))
	x1 = Variable(as_array(x.data + eps))
	y0 = f(x0)
	y1 = f(x1)
	return (y1.data - y0.data) / (2*eps)

# Square 태스트 함수
class SquareTest(unittest.TestCase):
	def test_forward(self):
		x = Variable(np.array(2.0))
	
		# Test Square
		y = square(x)
		expected = np.array(4.0)
		self.assertEqual(y.data, expected)
	
	def test_backward(self):
		x = Variable(np.array(3.0))
		y = square(x)
		y.backward()
		expected = np.array(6.0)
		self.assertEqual(x.grad, expected)
	
	def test_gradient_check(self):
		x = Variable(np.array(np.random.rand())) # 무작위 입력값 생성
		y = square(x)
		y.backward()
		num_grad = numerical_diff(square,x)
		flg = np.allclose(x.grad, num_grad) # 인스턴스 두 개가 가까운지 판정
		self.assertTrue(flg)

# python -m unittest FileName.py 실행 or
# unittest.main()
unittest.main()
