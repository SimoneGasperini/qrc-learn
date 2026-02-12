import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState


class LSTM(nn.Module):
    hidden_size: int
    out_size: int

    @nn.compact
    def __call__(self, x):
        cell = nn.LSTMCell(features=self.hidden_size)
        rnn = nn.RNN(cell=cell)
        h_seq = rnn(inputs=x)
        h_last = h_seq[:, -1, :]
        return nn.Dense(self.out_size)(h_last)


def make_windows(series, in_size, out_size):
    s = np.asarray(series, dtype=np.float32).reshape(-1)
    n = len(s) - in_size - out_size + 1
    x = np.stack([s[i : i + in_size] for i in range(n)], axis=0)[..., None]
    y = np.stack([s[i + in_size : i + in_size + out_size] for i in range(n)], axis=0)
    return x, y


def create_train_state(rng, model, optimizer, in_size):
    params = model.init(rng, jnp.ones((1, in_size, 1), dtype=jnp.float32))
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


def loss_fn(params, apply_fn, x_batch, y_batch):
    pred = apply_fn(params, x_batch)
    return jnp.mean((pred - y_batch) ** 2)


@jax.jit
def train_step(state, x_batch, y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(
        state.params, state.apply_fn, x_batch, y_batch
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, x_batch, y_batch):
    return loss_fn(state.params, state.apply_fn, x_batch, y_batch)
