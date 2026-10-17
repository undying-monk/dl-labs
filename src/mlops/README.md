### Run commands
```
$ python -m pytest src/mlops/tests/.. -v
$ fastapi dev --entrypoint src.mlops.app:app
```

We can config the entrypoint inside `pyproject.toml` config
```
[tool.fastapi]
entrypoint = "src.mlops.app:app"
```

### Container
```
$ uv pip freeze > requirements.txt
```

