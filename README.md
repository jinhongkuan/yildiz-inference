# About

Stochastic dynamics inference for social systems.

This repository contains scripts for executing Yildiz's Non-Parametric Differential Equation (NPDE) code on remote Modal hosts. Additionally, it includes data analysis scripts to process and interpret the results.

Linked:

- https://github.com/jinhongkuan/npde

# Instructions

## Authenticate with Modal

```
modal login
```

## Run the script

```
source venv/bin/activate
python -m modal run main.py
```

## Rebuilding the image

After updating the linked npde repo, modify the `echo` command in `main.py` to force a rebuild:

```
echo Rebuild npde version <version> # modify this to force a re-build after git repo is updated
```
