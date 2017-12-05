from t3f import decompositions


def add_n(tt_objects, max_tt_rank):
  """Adds a bunch of TT-object and round after each summation.

  This version implements a slow-to-compile but fast-to-execute (at least on
  a GPU) version: summing in a binary tree order.

  Args:
    tt_objects: a list of `TensorTrainBase` objects.
    max_tt_rank: a number, TT-rank for each individual rounding.

  Returns:
    Object of the same type as each input.
  """
  prev_level = tt_objects
  while len(prev_level) > 1:
    next_level = []
    for i in range(0, len(prev_level), 2):
      curr = prev_level[i]
      if i + 1 < len(prev_level):
        curr = decompositions.round(curr + prev_level[i + 1], max_tt_rank)
      next_level.append(curr)
    prev_level = next_level
  return prev_level[0]
