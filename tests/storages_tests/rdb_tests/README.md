### Migration test guide

1. Install Optuna

You can view the current stable schema version by checking `optuna/storages/_rdb/alembic/versions`.

```sh
> ls optuna/storages/_rdb/alembic/versions
v0.9.0.a.py  v1.2.0.a.py  v1.3.0.a.py  v2.4.0.a.py  v2.6.0.a_.py
```

I recommend you to create isolated environment using `venv` for this purpose.

```sh
> deactivate  # if you already use `venv` for development
> python3 -m venv venv_gen
> . venv_gen/bin/activate
> pip install optuna==2.6.0  # it depends on the output of `ls` above
```

2. Generate database

```sh
> python3 create_db.py
[I 2022-02-05 15:39:32,488] A new study created in RDB with name: single_empty
...
>
```

3. Switch Optuna version to the latest one

If you use `venv`, simply `deactivate` and re-activate your development environment.