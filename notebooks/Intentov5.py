import math,random,os

class Perceptron():

	def __init__(self):	
		self.perceptron = [] # array que nos define el perceptron y cada capa contiene un subarray de la forma: [numero neurons x capa,[umbrales de actuacion(u)],[pesos(w)],[salidas de las neuronas(a)]]
		self.ntrain = 0 #numero de entrenamientos ejecutados
		self.build() #construimos la red neuronal
		#self.train() #entrenamos la red neuronal
		while True: #utilizamos la red neuronal
			for i in range(self.perceptron[0][0]):
				self.perceptron[0][3][i] = input('entrada ' + str(i) + ' : ')
			self.percep()
	def build(self): #funcion que construye la red neuronal
		print('***Red Neuronal***')
		for i in range(4):
			capa = []
			u = []
			w = []
			a = []
			capa.append(4)#anadimos el numero de neuronas de la capa
			for j in range(capa[0]):
				u.append(random.random()) #creamos umbrales de actuacion aleatorios
				a.append(random.random()) #creamos salidas aleatorias
				wc = []
				if i != 0:
					for k in range(self.perceptron[i-1][0]):
						wc.append(random.random()) #creamos pesos aleatorios
				w.append(wc)
			capa.append(u)
			capa.append(w)
			capa.append(a)
			self.perceptron.append(capa) #anadimos la capa a la red neuronal


if __name__ == "__main__":
    Perceptron()
	



