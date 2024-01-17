# Estimating transformer memory consumption

I tried using formulas described in [a web
page](https://blog.eleuther.ai/transformer-math/) for estimating the memory
requirements of a transformer. The goal was to illustrate how long inputs make
it impossible to train transformers on a consumer GPU. Unfortunately the math
did not really agree with reality.

I tried loading up BigBird with classical attention and what should've fit did
not. And overall the values really underestimated the real memory cost.
