import sys
import genotypes
from graphviz import Digraph
import torch
import torch.nn as nn
from operators import *
from genotypes import *
from utils import *

def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)

def _parse_arch_(weight):
    s_n, e_n = 0, 0
    arch = []
    c_op = len(PRIMITIVES)
    i_op_none = PRIMITIVES.index('none')
    steps = 4
    for i_b in range(steps):
        s_n, e_n = e_n, e_n + i_b + 2
        W = weight[s_n:e_n].copy()
        i_ns = sorted(range(i_b+2), key=lambda i_n : -max(W[i_n][i_op] for i_op in range(c_op) if i_op != i_op_none))[:2]
        for i_n in i_ns:
            i_op_best = None
            for i_op in range(c_op):
                if i_op == i_op_none:
                    continue
                if i_op_best is None or W[i_n][i_op] > W[i_n][i_op_best]:
                    i_op_best = i_op
            arch.append((PRIMITIVES[i_op_best], i_n))
    return arch

if __name__ == '__main__':
    pth = torch.load('./trained_model/nas_arch.pkl')
    a_normal = pth['module.w_alpha_normal'].cpu().numpy()
    a_reduce = pth['module.w_alpha_reduction'].cpu().numpy()

    cell_norma = _parse_arch_(a_normal)
    cell_reduce = _parse_arch_(a_reduce)
    

    plot(cell_norma, 'normal')
    plot(cell_reduce, 'reduce')

