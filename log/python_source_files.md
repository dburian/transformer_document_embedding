# Python source files

I will need to write two types of code:

- official code of my model -- maybe uploaded to Hugging Face,
- code for my thesis -- creating tasks, comparing models.

This leads to having two python packages which I can easily reference without
worrying about import paths.

So let's place the code for my thesis under `src/` and the code of the model
under `model/` -- the submodule of `thesis_src`. Of course `thesis_src` will be
renamed to the model name, once I name it.

TODO: When I'm done with setting up the infrastructure, lets document it in
`thesis_src`'s readme and this repository's readme.
