# Configuration files

I dislike putting 100 arguments on a commandline so instead every script loads
up a single or more configuration files in yaml format. Each can have different
specification (which should be defined in `scripts.config_specs` along with
their description).

These configuration files are manipulated during the running of the script and
then another configuration file is saved in the root directory of the
experiment. This saved configuration file may slightly differ from the one that
was provided, but should completely describe what was done, such that the
experiment is repeatable.
