class Adam(Optimizer):
	def init(self, param):
		m = np.zeros_like(param)
		v = no.zeros_like(param)
		return m, v

	def update(self, step, grads, param, slots, opt_params):
		m , v = slots
		learning_rate, b1, b2, eps = opt_params
		m = (1 - b1) * grads + b1 * m # First moment estimate.
		v = (1 - b2) * (grads ** 2) + b2 * v # second moment estimate
		mhat = m / (1 - b1 ** (step + 1)) # bias correction
		vhat = v / (1 - b2 ** (step + 1))

		param = param - (
			learning_rate * mhat / (np.sqrt(vhat) + eps)).astype(param.dtype)

		return param, (m, v)		